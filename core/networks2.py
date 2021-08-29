import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.modules import SelfAttention
from torch.nn.utils import spectral_norm
from torchvision.models import vgg19
from torchvision.models.video import r2plus1d_18


class PoseRegressor2d1(nn.Module):
    """input is a frame"""
    def __init__(self,num_features):
        """
        convolutional encoder + linear layer at the end
        Returns:
            torch.tensor vecs:(b,16,3); tvec:(b,3); rvec:(b,3)
        """
        super(PoseRegressor2d1, self).__init__()
        self.num_features = num_features
        self.main = self.define_network(num_features)
        self.resnet= nn.Sequential(*list(vgg19(pretrained=True).children())[:-1])
        self.main1=nn.Sequential(nn.Conv2d(3, 32, 4,2,1),
                                 nn.BatchNorm2d(32),
                                 nn.LeakyReLU(inplace=True),
                                 nn.MaxPool2d(2,2), #64*64
                                 nn.Conv2d(32, 64, 4,2,1),
                                 nn.BatchNorm2d(64),
                                 nn.LeakyReLU(inplace=True),
                                 nn.MaxPool2d(2,2), #16,16
                                 nn.Conv2d(64, 128, 4,2,1),
                                 nn.BatchNorm2d(128),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Flatten(),
                                 nn.Linear(8*8*128,512),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(512,54)
                                )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.LeakyReLU(inplace=True),
            # 16 3d vectors, 3 angles (axis-angle repr) and 3 translation params --> 54
            nn.Linear(512, 54)
        )
        self.tanh=nn.Tanh()
    
    def define_network(self, num_features):
        layers = [self._add_conv_layer(in_ch=3, nf=num_features[0])]
        for i in range(1, len(num_features)):
            layers.append(self._add_conv_layer(num_features[i-1], num_features[i]))
            layers.append(self._add_down_layer2())
        layers.append(nn.Sequential(
            nn.Conv2d(num_features[-1], 256, 1, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512,54),  
#             nn.LeakyReLU(inplace=True)
        ))
        return nn.Sequential(*layers)
    
    def _add_conv_layer(self, in_ch, nf):
        return nn.Sequential(
            nn.Conv2d(in_ch, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(inplace=True)
        )
    
    def _add_down_layer(self, in_ch, nf):
        return nn.Sequential(
            nn.Conv2d(in_ch, nf, 3, 2, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(inplace=True)
        )
    def _add_down_layer2(self):
        return nn.Sequential(
            nn.MaxPool2d(2,stride=2)
        )

    def forward(self, input):
        out = self.resnet(input)
#         features = self.features(input)
        out=self.regressor(out)
        vecs = out[:, 0:48].view(-1, 16, 3)
        rvecs = out[:, 48:51].view(-1, 3)
        tvecs = out[:, 51:].view(-1, 3)
#         vecs = out[:, 0:48].reshape(-1, 16, 3)
#         rvecs = out[:, 48:51].reshape(-1, 3)
#         tvecs = out[:, 51:].reshape(-1, 3)

        return vecs, rvecs, tvecs


class MotionDiscriminator(nn.Module):

    def __init__(self,
                 linear_list,
                 num_layers=1,
                 output_size=2,
                 dropout=.0):
        super(MotionDiscriminator, self).__init__()
        self.linear_list = linear_list
        self.num_layers = num_layers
        layers=[]
        layers.append(spectral_norm(nn.Linear(48, linear_list[0])))
        layers.append(nn.LeakyReLU(inplace=True)) 
        layers.append(nn.Dropout(dropout))
        for i in range(len(linear_list)-1):
            layers.append(spectral_norm(nn.Linear(linear_list[i], linear_list[i+1])))
            layers.append(nn.LeakyReLU(inplace=True)),
            layers.append(nn.Dropout(dropout))
            
            
        self.perceptron=nn.Sequential(*layers)

        self.fc = spectral_norm(nn.Linear(linear_list[-1], output_size))

    def forward(self, sequence):
        """
        sequence: of shape [batch_size, input_size]
        
        """
        outputs=self.perceptron(sequence)

        output = self.fc(outputs)

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