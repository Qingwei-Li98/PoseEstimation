from torch.utils.data import Dataset
import os
import os.path
import random
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R
from PIL import Image 
import numpy as np
import torch
from tqdm import tqdm
from utils.helper_functions import get_file_num
from utils.dataset_helper import get_bbx,get_center_scale,get_affine_transform,affine_transform

hip = 'hip'
r_hip = 'r_hip'
r_knee = 'r_knee'
r_ankle = 'r_ankle'
r_ball = 'r_ball'
l_hip = 'l_hip'
l_knee = 'l_knee'
l_ankle = 'l_ankle'
l_ball = 'l_ball'
l_shoulder = 'l_shoulder'
l_elbow = 'l_elbow'
l_wrist = 'l_wrist'
l_thumb = 'l_thumb'
r_shoulder = 'r_shoulder'
r_elbow = 'r_elbow'
r_wrist = 'r_wrist'
r_thumb = 'r_thumb'
l_little = 'l_little'
r_little = 'r_little'

joint_indices = {
    l_hip: 6,
    r_hip: 1,
    l_shoulder: 17,
    r_shoulder: 25,
    l_knee: 7,
    r_knee: 2,
    l_elbow: 18, 
    r_elbow: 26,
    l_ankle: 8,
    r_ankle: 3,
    l_wrist: 19,
    r_wrist: 27,
    l_ball: 9,
    r_ball: 4,    
    l_little: 22,
    r_little: 30
}

joints=[]
for _,v in joint_indices.items():
    joints.append(v)
    
# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# def make_dataset(frame_root):
#     """
#         frame_root: the root path of frame folder, eg: /vol/biodata/data/human3.6m/training/S1/frames/
#     """
#     frame_paths = []
#     frame_names=[]
#     for subject in os.listdir(frame_root):        
#         activity=re.match(r'[a-zA-Z]+',subject).group()
#         video_frames=[]
#         d = os.path.join(frame_root, subject)
#         if not os.path.isdir(d):
#             continue
#         frame_names.append(subject)
#         for root, _, fnames in sorted(os.walk(d)):
#             for fname in fnames:
#                 if is_image_file(fname):
#                     path = os.path.join(root, fname)
#                     video_frames.append(path)
#         frame_paths.append(video_frames)

#     return frame_names,frame_paths

class Human36M(Dataset):
    def __init__(self, root,pose_root, activities=None,
             actors=None,path=None, transform=transforms.ToTensor(),is_train=True):
        """
            root: the root of data, eg: '/vol/biodata/data/human3.6m/training'
            acticities: the list of actavities, eg: ['directions', 'discussion', 'greeting', 'posing','waiting', 'walking']
            actors: list of actors, eg: ['S%d' % i for i in [1, 5, 6, 7, 8, 9]]
            sequence_len: the length of frame sequence
            transforms: torchvision.transforms to resize and crop frame to (256,256)
        """

        self.root = root
        self.pose_root=pose_root
        # load dataset
        self.sequences = []
        self.poses=[]
        self.transform=transform
        self.is_train=is_train
        self.dataset=None
        if path != None:
            self.load_seq(path)
        else:
            for actor in tqdm(actors): # actor is in the form of 'S1','S5'
                if not os.path.isdir(os.path.join(root, actor)):
                    continue
                frame_sequences = os.listdir(os.path.join(root, actor))
                for activity in activities:
                    activity_sequences = [s for s in frame_sequences if s.lower().startswith(activity.lower())]
                    for seq in activity_sequences:
                        bboxs=os.listdir(os.path.join(root, actor,seq))
                        for bbox in bboxs:
                            frames = os.listdir(os.path.join(root, actor,seq, bbox))
                            frames = [int(x[5:-4]) for x in frames] # like 'frame0001', take last 4 number
                            frames = sorted(frames) # sorted int value  
                            if len(frames)>1:
                                pose_path=os.path.join(pose_root,actor,seq)
                                if not os.path.isdir(pose_path):
                                    continue
                                pose_num=get_file_num(os.path.join(pose_root, actor,seq))
                                if pose_num<1:
                                    continue
                                for frame1_num in frames:
                                    frame2_num=random.sample(frames, 1)[0]
                                    pose_folder = os.path.join(self.pose_root, actor, seq)
                                    pose_name='2d_pose_'+str(frame1_num-1)+'.npy'
                                    pose_path=os.path.join(pose_folder,pose_name)
                                    if not os.path.exists(pose_path):
                                        print(os.path.join(pose_folder,pose_name))
                                        continue
                                    self.sequences.append({ 'actor': actor, # str
                                                       'activity_sequence': seq, # str
                                                       'bbox': bbox,
                                                       'frame1': frame1_num, # list of int
                                                       'frame2': frame2_num, # int
                                                          })

    
    def get_single(self, sequence):
        sequence_path=os.path.join(self.root, sequence['actor'], sequence['activity_sequence'],sequence['bbox'])
        frame_path=os.path.join(sequence_path,'frame{:0>4d}'.format( sequence['frame1'])+'.png')
        img = Image.open(frame_path)
        frame_tensor=self.transform(img) #(CHW)
        frame2_torch= self.get_frame2(sequence)
        pose2D=self.get_pose2D(sequence)

        return (frame_tensor,frame2_torch,pose2D)
    
    def get_frame2(self,sequence):
        frame2_num=sequence['frame2']
        sequence_path=os.path.join(self.root, sequence['actor'], sequence['activity_sequence'],sequence['bbox'])
        frame2_path = os.path.join(sequence_path,'frame{:0>4d}'.format(frame2_num)+'.png')
        frame2 = Image.open(frame2_path)
        frame2_size= np.array(frame2).shape #(H,W,C)
        frame2_torch=self.transform(frame2) 
        return frame2_torch
    
    def get_pose2D(self,sequence):  
        pose_folder = os.path.join(self.pose_root, sequence['actor'], sequence['activity_sequence'])
        pose_name='2d_pose_'+str(sequence['frame1']-1)+'.npy'
        pose_path=os.path.join(pose_folder,pose_name)
        pose=np.load(pose_path)
        pose=pose[joints,:]
        bbx=get_bbx(sequence['bbox'])
        center,scale=get_center_scale(bbx)
        trans = get_affine_transform(center, scale, 0,(256,256))
        for i in range(len(pose)):
            pose[i] = affine_transform(pose[i], trans)
        pose_tensor=torch.FloatTensor(pose)
        return pose_tensor

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
    
    def __getitem__(self, index):
        seq=self.sequences[index]
        return  self.get_single(seq)
        
    def __len__(self):
        if self.sequences:
            return len(self.sequences)
        else:
            raise(RuntimeError("Dataset is empty!"))
            return 0

        
                  

class Human36MPose(Dataset):
    def __init__(self,root,sequences):
        self.root=root
        self.sequences=sequences
        
    def get_single(self,sequence):
        pose_folder = os.path.join(self.root, sequence['actor'], sequence['activity_sequence'])
        pose_name='3d_pose_'+str(sequence['frame1']-1)+'.npy'
        pose_path=os.path.join(pose_folder,pose_name)
        pose=np.load(pose_path)
        pose_array = np.array(pose)
        pose = pose_array[joints,:]#(16, 3)
        pose = self.align(pose)
        first = pose
        pelvis = first[0] + (first[1] - first[0]) / 2
        pose -= pelvis
        pose = self.normalize(pose)
        pose_tensor = torch.FloatTensor(pose)
        
        return pose_tensor
    
    def align(self, points):
        """remove y component of hip line,
        align pelvis-neck line with z axis"""
        pelvis = points[0] + (points[1] - points[0]) / 2
        neck = points[2] + (points[3] - points[2]) / 2
        pelvis_neck_line = neck - pelvis
        # pelvis neck
        rot_axis1 = np.array([0, -1, 0])
        angle = self.get_angle(pelvis_neck_line, rot_axis1)
        cross_prod = np.cross(pelvis_neck_line, rot_axis1)
        cross_prod /= np.linalg.norm(cross_prod)
        R1 = R.from_rotvec(angle * cross_prod)
    
        points = R1.apply(points)
        # hip 
        hip_line = points[0] - points[1]
        
        rot_axis21 = np.array([-1, 0])
        rot_axis22 = np.array([1, 0])
        indices = np.array([0, 2])
        angle1 = self.get_angle(hip_line[indices], rot_axis21)
        angle2 = self.get_angle(hip_line[indices], rot_axis22)
        if angle1<angle2:
            angle=angle1
        else:
            angle=angle2
        R2 = R.from_rotvec(np.array([0, angle, 0]))
        points = R2.apply(points)

        rot_z = R.from_rotvec(np.array([0,0, np.pi]))
        rot_y = R.from_rotvec(np.array([0,np.pi, 0]))
        points = rot_y.apply(rot_z.apply(points))

        return points
    
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
    
    def get_angle(self,vec1, vec2):
        inv = vec1 @ vec2 / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return np.arccos(inv)
        
    def __getitem__(self,index):
        seq=self.sequences[index]
        return  self.get_single(seq)
        
    def __len__(self):
        if self.sequences:
            return len(self.sequences)
        else:
            raise(RuntimeError("Dataset is empty!"))
            return 0
        
        
        
        

                

    