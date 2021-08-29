import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.modules import SelfAttention
from torch.nn.utils import spectral_norm
from torchvision.models.video import r2plus1d_18


class PoseRegressor2d1(nn.Module):
    """input is sequence of 20 frames"""
    def __init__(self):
        super(PoseRegressor2d1, self).__init__()
        self.features = nn.Sequential(*list(r2plus1d_18(pretrained=True).children())[:-2])
        self.main = nn.Sequential(
            nn.Conv3d(512, 128, (3, 4, 4), (1, 2, 2), 1),
            nn.Flatten(),
            nn.Linear(128*3*8*8, 2048),
            nn.LeakyReLU(inplace=True),
            # 20 * 16 3d vectors for points, 3 rvec and 3 tvec = 57
            nn.Linear(2048, 54*20)
        )

    def forward(self, input):
        feats = self.features(input)
        out = self.main(feats)
        # vecs = out[:, 0:51*20].view(-1, 17, 3, 20)
        # rvec = out[:, 51*20:54*20].view(-1, 3, 20)
        # tvec = out[:, 54*20:].view(-1, 3, 20)
        vecs = out[:, 0:48*20].view(-1, 16, 3, 20)
        rvecs = out[:, 48*20:51*20].view(-1, 3, 20)
        tvecs = out[:, 51*20:].view(-1, 3, 20)

        return vecs, rvecs, tvecs


class MotionDiscriminator(nn.Module):

    def __init__(self,
                 rnn_size,
                 input_size,
                 num_layers,
                 output_size=2,
                 feature_pool="concat",
                 use_spectral_norm=False,
                 attention_size=1024,
                 attention_layers=1,
                 attention_dropout=0.5):

        super(MotionDiscriminator, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.feature_pool = feature_pool
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout

        self.gru = nn.GRU(self.input_size, self.rnn_size, num_layers=num_layers)

        linear_size = self.rnn_size if not feature_pool == "concat" else self.rnn_size * 2

        if feature_pool == "attention":
            self.attention = SelfAttention(attention_size=self.attention_size,
                                       layers=self.attention_layers,
                                       dropout=self.attention_dropout)
        if use_spectral_norm:
            self.fc = spectral_norm(nn.Linear(linear_size, output_size))
        else:
            self.fc = nn.Linear(linear_size, output_size)

    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, input_size]
        """
        batchsize, seqlen, input_size = sequence.shape
        sequence = torch.transpose(sequence, 0, 1)
        
        outputs, state = self.gru(sequence)

        if self.feature_pool == "concat":
            outputs = F.relu(outputs)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            output = self.fc(torch.cat([avg_pool, max_pool], dim=1))
        elif self.feature_pool == "attention":
            outputs = outputs.permute(1, 0, 2)
            y, attentions = self.attention(outputs)
            output = self.fc(y)
        else:
            output = self.fc(outputs[-1])

        return output


class ImageDecoder(nn.Module):
    def __init__(self, n_f, num_joints):
        super(ImageDecoder, self).__init__()
        """Input is 3x256x256 rgb image and 256x256 x num_joints transformed heatmaps"""

        self.main = nn.Sequential(
            nn.Conv2d(3 + num_joints, n_f, 3, 1, 1),
            nn.BatchNorm2d(n_f),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_f, n_f*2, 3, 2, 1),
            nn.BatchNorm2d(n_f*2),
            nn.LeakyReLU(inplace=True),
            # 128x128
            nn.Conv2d(n_f*2, n_f*2, 3, 1, 1),
            nn.BatchNorm2d(n_f*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_f*2, n_f * 4, 3, 2, 1),
            nn.BatchNorm2d(n_f * 4),
            nn.LeakyReLU(inplace=True),
            # 64x64
            nn.Conv2d(n_f * 4, n_f * 4, 3, 1, 1),
            nn.BatchNorm2d(n_f * 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_f * 4, n_f * 8, 3, 2, 1),
            # 32, 32
            nn.Conv2d(n_f * 8, n_f * 8, 3, 1, 1),
            nn.BatchNorm2d(n_f * 8),
            nn.LeakyReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.BatchNorm2d(n_f * 8),
            nn.LeakyReLU(inplace=True),
            # 64, 64

            nn.Conv2d(n_f * 8, n_f * 4, 3, 1, 1),
            nn.BatchNorm2d(n_f * 4),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # 128, 128
            nn.Conv2d(n_f*4, n_f, 3, 1, 1),
            nn.BatchNorm2d(n_f),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_f, n_f, 3, 1, 1),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            # 256, 256
            nn.Conv2d(n_f, n_f, 3, 1, 1),
            nn.BatchNorm2d(n_f),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_f, 3, 3, 1, 1)
        )

    def forward(self, image, heatmaps):

        return self.main(torch.cat([image, heatmaps], dim=1))