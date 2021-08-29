import os
import torch.nn as nn
import torch
import cv2
import numpy as np
import random
from PIL import Image 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import Adam

from core.utils.transforms import SpatialTransformer, CameraTransform
from core.utils.losses import limb_loss
from core.utils.model_template import Model
from core.utils.helper_functions import plot3d,plot2d
from core.networks2 import ImageDecoder, PoseRegressor2d1, MotionDiscriminator


class ForwardKinematics(Model):
    def __init__(self, train_dataset,val_ds, pose_dataset):
        super().__init__(train_dataset)

        self.train_dataset = train_dataset
        self.val_dataset=val_ds
        self.pose_dataset=pose_dataset
        train_pose_len=int(len(pose_dataset))-1000
        self.train_pose,self.val_pose=random_split(pose_dataset, 
                                                   [train_pose_len,1000], torch.Generator().manual_seed(0))

        print("(train_len,val_len)",len(train_dataset),len(val_ds))
        print("(train_pose_len)",train_pose_len)
        
        num_features=[32, 64, 128,128, 256]
        self.regressor_network = PoseRegressor2d1(num_features).to(self.device)
        self.decoder_network = ImageDecoder(n_f=32, num_joints=16).to(self.device)
        self.motion_discriminator_network = MotionDiscriminator(linear_list=[256,256,128], num_layers=2,
                                                                output_size=1,
                                                                dropout=0.2).to(self.device)

        self.cam_transform = CameraTransform(num_points=16, device=self.device)
        self.transformer = SpatialTransformer(device=self.device)

        self.regressor_optim = Adam(self.regressor_network.parameters(), lr=1e-5)
        self.decoder_optim = Adam(self.decoder_network.parameters(), lr=1e-5)
        self.discriminator_optim = Adam(self.motion_discriminator_network.parameters(), lr=5e-5)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear').to(self.device)

        torch.autograd.set_detect_anomaly(True)

    def train(self, batch_size, num_epochs, logdir, checkdir):
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.writer = SummaryWriter(logdir)
        
        train_dataloader = DataLoader(self.train_dataset, batch_size, shuffle=True, drop_last=True, num_workers=8)
        train_poseloader = DataLoader(self.train_pose, batch_size, shuffle=True, drop_last=True, num_workers=8)
        val_dataloader =  DataLoader(self.val_dataset, batch_size, shuffle=True, drop_last=True, num_workers=8)
        val_poseloader=  DataLoader(self.val_pose, batch_size, shuffle=True, drop_last=True, num_workers=8)
        
        l2_loss = nn.MSELoss()
        l1_loss=nn.SmoothL1Loss()

        pose_iterator = iter(train_poseloader)
        pose_iter = 0
        image_iter = 0
        num_iters = self.train_pose.__len__() // batch_size

        bce_loss = nn.BCEWithLogitsLoss()
        #loss = torch.Tensor([0.0]).float().cuda()

        real_labels = torch.full((batch_size,), 1, dtype=torch.float32, device=self.device)
        fake_labels = torch.full((batch_size,), 0, dtype=torch.float32, device=self.device)

        ones = torch.ones(batch_size, 3).to(self.device)
        ones[:, 2] = -1
        best_loss = float("inf")
        val_loss=0

        for i in range(num_epochs):
            print('epoch:',i)
            if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            for batch in train_dataloader:
                # restart loop over pose samples (and shuffle)
                if pose_iter == num_iters:
                    pose_iterator = iter(train_poseloader)
                    pose_iter = 0
                self.regressor_network.train()
                self.decoder_network.train()
                self.motion_discriminator_network.train()
                
                self.regressor_network.zero_grad()
                self.decoder_network.zero_grad()
                self.motion_discriminator_network.zero_grad()

                frame1, frame2, preds2d = batch
                frame1, frame2, preds2d = frame1.to(self.device), frame2.to(self.device), preds2d.to(self.device)
                preds2d = self.normalize(preds2d).float()

                real_poses = next(pose_iterator).to(self.device).float()
                # error view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
                real_poses = real_poses.reshape(-1, 16*3)

                pose_vectors, rvecs, tvecs = self.regressor_network(frame1)

                # make sure that z of tvec points along negative z-axis
                tvecs = torch.abs(tvecs) * ones - 1

                # loop over all timepoints
            
                cam3d, proj = self.cam_transform(pose_vectors, rvecs, tvecs)
                projected=proj

                cam3d = cam3d[..., 0]

                # transform back
                projected1 = (projected+1)*32

                projected1 = projected1.reshape(-1, 16, 2)

                heatmaps = self.transformer(projected1.flip(-1))

                heatmaps = self.upsample(heatmaps)

                # convert batch*T, ch, h, w to batch, ch, h, w, T for visualization

                heatmaps1 = heatmaps.view(batch_size, 16, 256, 256)

                recon = self.decoder_network(frame2, heatmaps)

#                 transform back and convert batch*T to batch, ch, h, w
                recon = recon.view(batch_size, 3, 256, 256)

#                 update discriminator
                fake_points = pose_vectors.view(-1, 16*3)
                d_real = self.motion_discriminator_network(real_poses).view(batch_size)
                d_fake = self.motion_discriminator_network(fake_points.detach()).view(batch_size)

                d_real_loss = bce_loss(d_real, real_labels)
                d_fake_loss = bce_loss(d_fake, fake_labels)

                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()

                self.discriminator_optim.step()

                d_fake1 = self.motion_discriminator_network(fake_points).view(batch_size)

                adv_loss = bce_loss(d_fake1, real_labels)

                recon_loss = l2_loss(frame1, recon)

                loss2d = l2_loss(projected, preds2d)
#                 l_loss = limb_loss(pose_vectors)

                loss = recon_loss + 10. * loss2d  +  0.2 * adv_loss #+0.2 * l_loss 
#                 loss = 10*loss2d+0.1*l_loss

                loss.backward()
                
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                self.regressor_optim.step()
                self.decoder_optim.step()

                image_iter += 1
                pose_iter += 1
                if image_iter%250==0:
                    print(image_iter)
                
                self.writer.add_scalar('adversarial loss', adv_loss, image_iter)
                self.writer.add_scalar('reconstruction loss', recon_loss, image_iter)
                self.writer.add_scalar('2d loss', loss2d, image_iter)
#                 self.writer.add_scalar('limb loss', l_loss, image_iter)
                self.writer.add_scalar('total loss', loss, image_iter)

                if image_iter % 500 == 0:
                    
                    self.write_tb(logdir, frame1,frame2,cam3d,projected,preds2d,recon,heatmaps1,image_iter)

                    
                if image_iter % 5000 == 0:
                    chk_path = checkdir
                    if not os.path.isdir(chk_path):
                        os.makedirs(chk_path)
                    self.save_checkpoint(path=chk_path, epoch=str(image_iter/5000))
            
            val_loss=self.test(batch_size,val_dataloader, val_poseloader,logdir,image_iter)
            chk_path = checkdir
            if not os.path.isdir(chk_path):
                os.makedirs(chk_path)
            self.save_checkpoint(path=chk_path, epoch='epoch'+str(i))
        print('Training finished')
                
     
    def write_tb(self, logdir, frame1,frame2,cam3d,projected,preds2d,recon,heatmaps1,image_iter):
        input_grid1 = self.show_images(frame1.detach())
        input_grid2 = self.show_images(frame2.detach())
        recon_grid = self.show_images(recon.detach())
        heatmap_grid = self.show_images(heatmaps1.sum(1, keepdims=True).detach())

        # 3d plots
        plot_path = os.path.join(logdir, "3d_plots")
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        plot3d(cam3d, path=os.path.join(plot_path, 'plot.png'))
        # 2d plots
        plot2d_path = os.path.join(logdir, "2d_plots")
        if not os.path.exists(plot2d_path):
            os.makedirs(plot2d_path)
        plot2d(projected,path=os.path.join(plot2d_path, 'plot.png'))
        #2d gt plot
        plot2dgt_path = os.path.join(logdir, "2d_gt")
        if not os.path.exists(plot2dgt_path):
            os.makedirs(plot2dgt_path)
        plot2d(preds2d,path=os.path.join(plot2dgt_path, 'plot.png'))


        # load image
        plot_img = cv2.imread(os.path.join(plot_path, 'plot.png'))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
        plot_img = torch.from_numpy(plot_img).permute(2, 0, 1).unsqueeze(0)
        plot_grid = self.show_images(plot_img)

        plot_img2d = cv2.imread(os.path.join(plot2d_path, 'plot.png'))
        plot_img2d = cv2.cvtColor(plot_img2d, cv2.COLOR_BGR2RGB)
        plot_img2d = torch.from_numpy(plot_img2d).permute(2, 0, 1).unsqueeze(0)
        plot_grid2d = self.show_images(plot_img2d)

        plot_2dgt = cv2.imread(os.path.join(plot2dgt_path, 'plot.png'))
        plot_2dgt = cv2.cvtColor(plot_2dgt, cv2.COLOR_BGR2RGB)
        plot_2dgt = torch.from_numpy(plot_2dgt).permute(2, 0, 1).unsqueeze(0)
        plot_grid2dgt = self.show_images(plot_2dgt)

        self.writer.add_image('reconstructed', recon_grid, image_iter)
        self.writer.add_image('heatmaps', heatmap_grid, image_iter)
        self.writer.add_image('input1', input_grid1, image_iter)
        self.writer.add_image('input2', input_grid2, image_iter)
        self.writer.add_image('3d plot', plot_grid, image_iter)
        self.writer.add_image('2d plot', plot_grid2d, image_iter)
        self.writer.add_image('2d ground truth', plot_grid2dgt, image_iter)
        
         
    def write_outlier(self, logdir, frame1,frame2,cam3d,projected,preds2d,recon,heatmaps1,image_iter):
        input_grid1 = self.show_images(frame1.detach())
        input_grid2 = self.show_images(frame2.detach())
        recon_grid = self.show_images(recon.detach())
        heatmap_grid = self.show_images(heatmaps1.sum(1, keepdims=True).detach())

        # 3d plots
        plot_path = os.path.join(logdir, "3d_outliers")
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        plot3d(cam3d, path=os.path.join(plot_path, 'plot.png'))
        # 2d plots
        plot2d_path = os.path.join(logdir, "2d_outliers")
        if not os.path.exists(plot2d_path):
            os.makedirs(plot2d_path)
        plot2d(projected,path=os.path.join(plot2d_path, 'plot.png'))
        #2d gt plot
        plot2dgt_path = os.path.join(logdir, "2d_gt_outliers")
        if not os.path.exists(plot2dgt_path):
            os.makedirs(plot2dgt_path)
        plot2d(preds2d,path=os.path.join(plot2dgt_path, 'plot.png'))


        # load image
        plot_img = cv2.imread(os.path.join(plot_path, 'plot.png'))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
        plot_img = torch.from_numpy(plot_img).permute(2, 0, 1).unsqueeze(0)
        plot_grid = self.show_images(plot_img)

        plot_img2d = cv2.imread(os.path.join(plot2d_path, 'plot.png'))
        plot_img2d = cv2.cvtColor(plot_img2d, cv2.COLOR_BGR2RGB)
        plot_img2d = torch.from_numpy(plot_img2d).permute(2, 0, 1).unsqueeze(0)
        plot_grid2d = self.show_images(plot_img2d)

        plot_2dgt = cv2.imread(os.path.join(plot2dgt_path, 'plot.png'))
        plot_2dgt = cv2.cvtColor(plot_2dgt, cv2.COLOR_BGR2RGB)
        plot_2dgt = torch.from_numpy(plot_2dgt).permute(2, 0, 1).unsqueeze(0)
        plot_grid2dgt = self.show_images(plot_2dgt)

        self.writer.add_image('reconstructed_outliers', recon_grid, image_iter)
        self.writer.add_image('heatmaps_outliers', heatmap_grid, image_iter)
        self.writer.add_image('input1_outliers', input_grid1, image_iter)
        self.writer.add_image('input2_outliers', input_grid2, image_iter)
        self.writer.add_image('3d outliers', plot_grid, image_iter)
        self.writer.add_image('2d outliers', plot_grid2d, image_iter)
        self.writer.add_image('2d ground truth of outliers', plot_grid2dgt, image_iter)

    def test(self, batch_size, val_dataloader,val_poseloader,logdir,train_iter):
        self.regressor_network.eval()
        self.decoder_network.eval()
        self.motion_discriminator_network.eval()
        with torch.no_grad():
            val_loss = 0
            adv_loss=0
            recon_loss=0
            loss2d=0
#             l_loss=0
            img_iter=0
            pose_iter = 0
            num_iters = val_poseloader.__len__() // batch_size
            real_labels = torch.full((batch_size,), 1, dtype=torch.float32, device=self.device)
            fake_labels = torch.full((batch_size,), 0, dtype=torch.float32, device=self.device)
            ones = torch.ones(batch_size, 3).to(self.device)
            ones[:, 2] = -1
            pose_iterator = iter(val_poseloader)
            bce_loss = nn.BCEWithLogitsLoss()
            l2_loss = nn.MSELoss()
            for batch in val_dataloader:
                if pose_iter == num_iters:
                    pose_iterator = iter(val_poseloader)
                    pose_iter = 0
                    
                frame1, frame2, preds2d = batch
                frame1, frame2, preds2d = frame1.to(self.device), frame2.to(self.device), preds2d.to(self.device)
                preds2d = self.normalize(preds2d).float()
                real_poses = next(pose_iterator).to(self.device).float()
                real_poses = real_poses.reshape(-1, 16*3)
                pose_vectors, rvecs, tvecs = self.regressor_network(frame1)
                tvecs = torch.abs(tvecs) * ones - 1
                cam3d, proj = self.cam_transform(pose_vectors, rvecs, tvecs)
                cam3d = cam3d[..., 0]
                projected1 = (proj+1)*32
                projected1 = projected1.reshape(-1, 16, 2)
                heatmaps = self.transformer(projected1.flip(-1))
                heatmaps = self.upsample(heatmaps)
                recon = self.decoder_network(frame2, heatmaps)
                fake_points = pose_vectors.view(-1, 16*3)
                d_real = self.motion_discriminator_network(real_poses).view(batch_size)
                d_fake = self.motion_discriminator_network(fake_points).view(batch_size)

                d_real_loss = bce_loss(d_real, real_labels).item()
                d_fake_loss = bce_loss(d_fake, fake_labels).item()

                d_loss = d_real_loss + d_fake_loss
                adv_loss += bce_loss(d_fake, real_labels).item()

                recon_loss += l2_loss(frame1, recon).item()

                loss2d += l2_loss(proj, preds2d).item()
#                 l_loss += limb_loss(pose_vectors).item()
                img_iter += 1
                pose_iter += 1
        
            adv_loss=adv_loss/img_iter
            recon_loss=recon_loss/img_iter
            loss2d=loss2d/img_iter
#             l_loss=l_loss/img_iter
            val_loss=loss2d
            val_loss = recon_loss + 10. * loss2d + 0.2 * adv_loss #  + 0.2 * l_loss  
            print('val_loss:',val_loss,'loss2d',loss2d,'recon_loss',recon_loss,'adv_loss',adv_loss)
            self.writer.add_scalar('val: adversarial loss', adv_loss, train_iter)
            self.writer.add_scalar('val: reconstruction loss', recon_loss, train_iter)
            self.writer.add_scalar('val:2d loss', loss2d,train_iter)
#             self.writer.add_scalar('val:limb loss', l_loss, train_iter)
            self.writer.add_scalar('val:total loss', val_loss, train_iter)
            if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            return val_loss
            
                    
    
    
    def normalize(self, points):
        # normalize to -1 to 1
        points = points/256
        points = (points*2) - 1

        return points

    def load_pretrained_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.regressor_network.load_state_dict(checkpoint["regressor_network"])
        self.decoder_network.load_state_dict(checkpoint["decoder_network"])
#         self.motion_discriminator_network.load_state_dict(checkpoint["motion_discriminator_network"])
        print("load pretrained checkpoint successfully")

    def predict(self, img):
        self.regressor_network.eval()
        with torch.no_grad():
            ones = torch.ones(img.size()[0], 3).to(self.device)
            ones[:, 2] = -1

            points3d, rvecs, tvecs = self.regressor_network(img)

            tvecs = torch.abs(tvecs) * ones - 1
            cam3d, projected = self.cam_transform(points3d, rvecs, tvecs)
            projected = (projected+1)*128
                
        return cam3d[...,0], projected,tvecs










