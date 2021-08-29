import pickle
import os
import sys
import numpy as np
import json_tricks as json
import cv2
import torch
import copy
import random
import torchvision.transforms as transforms
from glob import glob
from tqdm import tqdm
from PIL import Image 
from natsort import natsorted
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from utils.dataset_helper import get_affine_transform,affine_transform
from core.utils.helper_functions import load_matfile

class InfantDataSet(Dataset):
    def __init__(self, root,pose_root,subjects,path=None, transform=transforms.ToTensor(),is_train=True):
        """
            root: the root of data, eg: '/vol/biodata/data/human3.6m/training'
            pose_root: the root of 2d pose;
            subjects: subjects=['EMT4', 'EMT7', 'EMT38', 'EMT36', 'EMT31', 'EMT43',
                        'EMT5', 'EMT9', 'EMT47', 'EMT45', 'EMT29', 'EMT42',
                        'EMT23', 'EMT41', 'EMT37', 'EMT48', 'EMT44', 'EMT46',
                        'EMT20', 'EMT34', 'EMT11', 'EMT30', 'EMT39', 'EMT35',
                        'EMT14']
            transforms: torchvision.transforms to resize and crop frame to (256,256)
        """

        self.root = root
        self.pose_root=pose_root
        self.indices = np.array([3, 2, 9, 8, 4, 1, 10, 7, 13, 12, 15, 14, 5, 0, 11, 6])
        self.subjects=subjects
        # load dataset
        self.sequences = []
        self.transform=transform
        self.is_train=is_train
        self.dataset=None
        if path != None:
            self.load_seq(path)
        else:
            for subject in tqdm(self.subjects): # actor is in the form of 'S1','S5'
                if not os.path.isdir(os.path.join(root, subject)):
                    print(subject,"not exists")
                    continue
                frames = natsorted(os.listdir(os.path.join(root, subject)))
                frame_nums = [int(x[6:-4]) for x in frames]
                frame_nums=np.array(sorted(frame_nums))-1
                pred_path=os.path.join(pose_root,subject+'.pickle')
                if not os.path.exists(pred_path):
                    print(pred_path,"not exists")
                    continue
                with open(pred_path, "rb") as file:
                    pickle_file = pickle.load(file)
                for frame in frames:
                    frame_num=self.find_index(frame,pickle_file)
                    frame2=random.sample(frames, 1)[0]
                    seq={'subject': subject,
                         'bbox':pickle_file[frame_num]['bounding_box'],
                         'center': pickle_file[frame_num]['center'],
                         'scale': pickle_file[frame_num]['scale'],
                         'frame_num':frame_num,
                         'frame1':pickle_file[frame_num]['frame_id'],
                         'frame2': frame2,
                         'pose_2d':pickle_file[frame_num]['predicted_keypoints']
                        }
                    self.sequences.append(seq)

    
    def get_single(self, sequence):
        bbox=sequence['bbox']
        center=sequence['center']
        scale=np.asarray([sequence['scale'],sequence['scale']])
        image_size=(256,256)
        frame1_path=os.path.join(self.root,sequence['subject'],sequence['frame1'])
        frame2_path=os.path.join(self.root,sequence['subject'],sequence['frame2'])
        pose_2d=sequence['pose_2d'][self.indices]
        frame1=Image.open(frame1_path)
        frame2=Image.open(frame2_path)
        trans = get_affine_transform(center, scale, 0, image_size)
        frame1=cv2.warpAffine(np.array(frame1), trans, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
        frame2=cv2.warpAffine(np.array(frame2), trans, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
        frame1_tensor=self.transform(frame1)
        frame2_tensor=self.transform(frame2)
        pose2d_tensor=torch.FloatTensor(pose_2d)
        return frame1_tensor,frame2_tensor,pose2d_tensor
    
    def find_index(self,s,file):
        j=None
        for i in range(len(file)):
            if s == file[i]['frame_id']:
                j=i
                return j

    def get_sequences(self):
        return self.sequences
    
    def save_seq(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        seq=self.__dict__['sequences']
        torch.save(seq, os.path.join(path,'sequences.tar'))
        print("saved successfully!")
        
    def load_seq(self,path):
        seq_load=torch.load(path)
        self.sequences=seq_load
        print("load successfully!")
    
    def __getitem__(self, index):
        seq=self.sequences[index]
        return  self.get_single(seq)
        
    def __len__(self):
        return len(self.sequences)

class Infant3DPose(Dataset):
    def __init__(self, path,load_path=None):
        self.path = path
        self.indices = np.array([15, 14, 10, 6, 3, 0, 11, 7, 4, 1, 12, 8, 5, 2, 13, 9])
        self.keypoints = ['hips', 'shoulders', 'knees', 'elbows', 'ankles', 'wrists', 'feet', 'hands']
        self.sequences=[]
        if load_path != None:
            self.load_seq(load_path)
        else:
            self.sequences=self.get_sequences()
            
    def save_seq(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        seq=self.__dict__['sequences']
        torch.save(seq, os.path.join(path,'pose3d_sequences.tar'))
        print("saved successfully!")
        
    def load_seq(self,path):
        seq_load=torch.load(path)
        self.sequences=seq_load
        print("load successfully!")

    def get_sequences(self):
        spatial_indices = np.array([1, 2, 0])
        sequences = []
        files = glob(os.path.join(self.path, '*.mat'))
        sequences=[]
        for file in tqdm(files):
            # subj = os.path.splitext(os.path.split(file)[-1])[0]
            data, timestamps = load_matfile(file)
            # only xyz
            data = data[:, 0:3]
            data = data.transpose((0, 2, 1))
            # tracker data is 120 Hz, camera is 30 Hz, so factor is 4
            tracker_data = data
            for i in range(len(tracker_data)):
                pose = tracker_data[i]
                # change keypoint order and spatial orientation
                pose = pose[self.indices]
                pose = pose[..., spatial_indices]
                pose = self.align(pose)
                first = pose
                pelvis = first[0] + (first[1] - first[0]) / 2
                pose -= pelvis
                pose = self.normalize(pose)
                pose1 = np.copy(pose)
                pose1[..., 1] = pose[..., 2]
                pose1[..., 2] = pose[..., 1]
                sequences.append(pose1)
        return sequences

    def normalize(self, points):
        first = points
        # normalize to unit cube -1 to 1
        max_ = np.abs(first.max())
        min_ = np.abs(first.min())
        if max_ >= min_:
            points /= max_
        else:
            points /= min_
        return points

    def get_angle(self, vec1, vec2):
        inv = vec1 @ vec2 / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return np.arccos(inv)

    def align(self, sequence):
        """remove y component of hip line,
        align pelvis-neck line with z axis"""
        points = sequence
        hip_line = points[0] - points[1]
        pelvis = points[0] + (points[1] - points[0]) / 2
        neck = points[2] + (points[3] - points[2]) / 2 
        pelvis_neck_line = neck - pelvis

        # pelvis neck
        rot_axis1 = np.array([0, 0, 1])
        angle = self.get_angle(pelvis_neck_line, rot_axis1)
        cross_prod = np.cross(pelvis_neck_line, rot_axis1)
        cross_prod /= np.linalg.norm(cross_prod)
        R1 = R.from_rotvec(angle * cross_prod)
        points = R1.apply(points)
        # hip 
        hip_line = points[0] - points[1]
        rot_axis2 = np.array([1, 0])
        angle = self.get_angle(hip_line[0:2], rot_axis2)
        cross_prod = np.cross(hip_line[0:2], rot_axis2)
        cross_prod /= np.linalg.norm(cross_prod)
        R2 = R.from_rotvec(angle * np.array([0, 0, cross_prod]))
        points=R2.apply(points)
        rot = R.from_rotvec(np.array([0., 0., np.pi]))
        points=rot.apply(points)
        return points

    def __getitem__(self, idx):
        pose = np.array(self.sequences[idx])
#         p1 = np.copy(pose)
#         p1[..., 1] = pose[..., 2]
#         p1[..., 2] = pose[..., 1]
        pose_tensor=torch.FloatTensor(pose)

        return pose_tensor

    def __len__(self):
        return len(self.sequences)
