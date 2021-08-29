import os
import torch.nn as nn
import torch
import cv2
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam

from core.utils.transforms import SpatialTransformer, CameraTransform
from core.utils.losses import limb_loss
from core.utils.model_template import Model
from core.utils.helper_functions import plot3d
from core.networks import ImageDecoder, PoseRegressor2d1, MotionDiscriminator


class ForwardKinematics(Model):
    def __init__(self, dataset, pose_dataset):
        super().__init__(dataset)

        self.pose_dataset = pose_dataset

        self.regressor_network = PoseRegressor2d1().to(self.device)
        self.decoder_network = ImageDecoder(n_f=32, num_joints=16).to(self.device)
        self.motion_discriminator_network = MotionDiscriminator(rnn_size=512, input_size=16*3, num_layers=1, output_size=1,
                                                        feature_pool='attention', use_spectral_norm=True,attention_size=512,
                                                        attention_dropout=0.2).to(self.device)

        self.cam_transform = CameraTransform(num_points=16, device=self.device)
        self.transformer = SpatialTransformer(device=self.device)

        self.regressor_optim = Adam(self.regressor_network.parameters(), lr=1e-4)
        self.decoder_optim = Adam(self.decoder_network.parameters(), lr=1e-4)
        self.discriminator_optim = Adam(self.motion_discriminator_network.parameters(), lr=1e-4)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear').to(self.device)

        torch.autograd.set_detect_anomaly(True)

    def train(self, batch_size, num_epochs, logdir, checkdir,len_seq=20):
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.writer = SummaryWriter(logdir)
        
        dataloader = DataLoader(self.dataset, batch_size, shuffle=True, drop_last=True, num_workers=8)
        pose_dataloader = DataLoader(self.pose_dataset, batch_size, shuffle=True, drop_last=True, num_workers=8)

        l2_loss = nn.MSELoss()

        pose_iterator = iter(pose_dataloader)
        pose_iter = 0
        image_iter = 0
        num_iters = self.pose_dataset.__len__() // batch_size

        bce_loss = nn.BCEWithLogitsLoss()
        #loss = torch.Tensor([0.0]).float().cuda()


        real_labels = torch.full((batch_size,), 1, dtype=torch.float32, device=self.device)
        fake_labels = torch.full((batch_size,), 0, dtype=torch.float32, device=self.device)

        ones = torch.ones(batch_size, 3, len_seq).to(self.device)
        ones[:, 2] = -1
        self.regressor_network.train()

        for i in range(num_epochs):
            print('epoch:',i)
            if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            for batch in dataloader:
                # restart loop over pose samples (and shuffle)
                if pose_iter == num_iters:
                    pose_iterator = iter(pose_dataloader)
                    pose_iter = 0
                self.regressor_network.zero_grad()
                self.decoder_network.zero_grad()
                self.motion_discriminator_network.zero_grad()

                frames, frame2, preds2d = batch
                frames, frame2, preds2d = frames.to(self.device), frame2.to(self.device), preds2d.to(self.device)
                preds2d = self.normalize(preds2d).float()

                real_poses = next(pose_iterator).to(self.device).float()
                # error view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
                #real_poses = real_poses.permute(0, 3, 1, 2).view(-1, 20, 16*3) 
                real_poses = real_poses.permute(0, 3, 1, 2).reshape(-1, len_seq, 16*3)

                #frames = frames.permute(0, 1, 4, 2, 3) #(batch, channel, seq_len,h,w)

                pose_vectors, rvecs, tvecs = self.regressor_network(frames)

                # make sure that z of tvec points along negative z-axis
                tvecs = torch.abs(tvecs) * ones - 1

                # loop over all timepoints
                projected = []
                cam3d_list = []
                for i in range(len_seq):
                    cam3d, proj = self.cam_transform(pose_vectors[..., i], rvecs[..., i], tvecs[..., i])
                    projected.append(proj)
                    cam3d_list.append(cam3d)

                projected = torch.stack(projected, dim=-1)
                cam3d = torch.stack(cam3d_list, dim=-1)

                cam3d = cam3d[..., 0, 0]

                # transform back
                projected1 = (projected+1)*32

                projected1 = projected1.permute(0, 3, 1, 2).reshape(-1, 16, 2)

                heatmaps = self.transformer(projected1.flip(-1))

                heatmaps = self.upsample(heatmaps)

                # convert batch*T, ch, h, w to batch, ch, h, w, T for visualization

                heatmaps1 = heatmaps.view(batch_size, len_seq, 16, 256, 256).permute(0, 2, 3, 4, 1)

                # replicate frame
                frame2 = frame2.unsqueeze(1)
                frame2 = frame2.repeat(1, len_seq, 1, 1, 1).view(-1, 3, 256, 256)

                recon = self.decoder_network(frame2, heatmaps)

                # transform back and convert batch*T to batch, ch, T, h, w
                recon = recon.view(batch_size,len_seq, 3, 256, 256).permute(0, 2, 1, 3, 4)

                # update discriminator
                fake_points = pose_vectors.permute(0, 3, 1, 2).view(-1, len_seq, 16*3)
                d_real = self.motion_discriminator_network(real_poses).view(batch_size)
                d_fake = self.motion_discriminator_network(fake_points.detach()).view(batch_size)

                d_real_loss = bce_loss(d_real, real_labels)
                d_fake_loss = bce_loss(d_fake, fake_labels)

                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()

                self.discriminator_optim.step()

                d_fake1 = self.motion_discriminator_network(fake_points).view(batch_size)

                adv_loss = bce_loss(d_fake1, real_labels)

                recon_loss = l2_loss(frames, recon)
                
                loss2d = l2_loss(projected, preds2d)

                l_loss = limb_loss(pose_vectors)

                loss = recon_loss + 10. * loss2d  + 0.2 * l_loss  + 0.2 * adv_loss

                loss.backward()
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                self.regressor_optim.step()
                self.decoder_optim.step()

                image_iter += 1
                pose_iter += 1
                if image_iter%50==0:
                    print(image_iter)
                
                self.writer.add_scalar('adversarial loss', adv_loss, image_iter)
                self.writer.add_scalar('reconstruction loss', recon_loss, image_iter)
                self.writer.add_scalar('2d loss', loss2d, image_iter)
                self.writer.add_scalar('limb loss', l_loss, image_iter)

                if image_iter % 500 == 0:
                    input_grid1 = self.show_sequence_images(frames.permute(0, 1, 3, 4, 2).detach())
                    input_grid2 = self.show_images(frame2.detach())
                    recon_grid = self.show_sequence_images(recon.permute(0, 1, 3, 4, 2).detach())
                    heatmap_grid = self.show_sequence_images(heatmaps1.sum(1, keepdims=True).detach())

                    # 3d plots
                    plot_path = os.path.join(logdir, "3d_plots")
                    if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    plot3d(cam3d, path=os.path.join(plot_path, 'plot.png'))

                    # load image
                    plot_img = cv2.imread(os.path.join(plot_path, 'plot.png'))
                    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
                    plot_img = torch.from_numpy(plot_img).permute(2, 0, 1).unsqueeze(0)
                    plot_grid = self.show_images(plot_img)

                    self.writer.add_image('reconstructed', recon_grid, image_iter)
                    self.writer.add_image('heatmaps', heatmap_grid, image_iter)
                    self.writer.add_image('input1', input_grid1, image_iter)
                    self.writer.add_image('input2', input_grid2, image_iter)
                    self.writer.add_image('3d plot', plot_grid, image_iter)
                    
                if image_iter % 5000 == 0:
                    chk_path = checkdir
                    if not os.path.isdir(chk_path):
                        os.makedirs(chk_path)
                    self.save_checkpoint(path=chk_path, epoch=str(image_iter))
            chk_path = checkdir
            if not os.path.isdir(chk_path):
                os.makedirs(chk_path)
                self.save_checkpoint(path=chk_path, epoch=str(i))
            

    def normalize(self, points):
        # normalize to -1 to 1
        points = points/256
        points = (points*2) - 1

        return points

    def load_pretrained_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.regressor_network.load_state_dict(checkpoint["regressor_network"])
        self.decoder_network.load_state_dict(checkpoint["decoder_network"])

    def predict(self, img):
        self.regressor_network.eval()
        vecs, rvecs, tvecs = self.regressor_network(img)

        points3d, rvecs, tvecs = self.regressor_network(vecs)

        projected = self.cam_transform(points3d, rvecs, tvecs)

        projected = (projected + 1) * 128

        return points3d, projected










