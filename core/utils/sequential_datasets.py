import pickle
import os
import numpy as np
import json_tricks as json
import cv2
import torch
import copy
import random
from glob import glob
from natsort import natsorted
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from core.utils.transforms import get_affine_transform
from core.utils.helper_functions import load_matfile

"""
default keypoint order (left, right):
0 1 hips
2 3 shoulders
4 5 knees
6 7 elbows
8 9 ankles
10 11 wrists
12 13 feet
14 15 hands
"""


class SequentialInfantDataset(Dataset):
    def __init__(self, img_path, meta_path, pred_path, flow_path, flow_threshold, seq_length, sampling, is_train, split,
                 transform=None, return_img=False):
        self.is_train = is_train
        self.img_path = img_path
        self.image_dict = self.get_img_dict()
        with open(pred_path, "rb") as file:
            self.predictions = pickle.load(file)
        with open(meta_path) as file:
            self.center_dict = json.load(file)
        self.flow_threshold = flow_threshold
        self.indices = np.array([3, 2, 9, 8, 4, 1, 10, 7, 13, 12, 15, 14, 5, 0, 11, 6])

        self.seq_length = seq_length
        self.sampling = sampling
        self.scale_factor = 0.2
        self.rotation_factor = 0
        self.flip = False
        self.split = split
        self.return_img = return_img

        self.subjects = np.asarray(['EMT4', 'EMT7', 'EMT38', 'EMT36', 'EMT31', 'EMT43',
                                    'EMT5', 'EMT9', 'EMT47', 'EMT45', 'EMT29', 'EMT42',
                                    'EMT23', 'EMT41', 'EMT37', 'EMT48', 'EMT44', 'EMT46',
                                    'EMT20', 'EMT34', 'EMT11', 'EMT30', 'EMT39', 'EMT35',
                                    'EMT14'])

        self.splits = [tuple(np.arange(i, i + 6)) for i in range(0, 24, 6)]
        self.splits = [[i for i in split] for split in self.splits]

        self.test_ids = self.subjects[self.splits[split]]

        self.image_size = np.asarray((256, 256))

        self.transform = transform
        with open(flow_path, 'r') as file:
            self.flows = json.load(file)

        self.db = self._get_db()

    def _get_db(self):
        db = []
        folders = glob(os.path.join(self.img_path, '*/'))
        for folder in folders:
            subj_id = folder.split('/')[-2]
            if self.is_train:
                if subj_id not in self.test_ids and subj_id != 'EMT36':
                    frames = natsorted(glob(os.path.join(folder, '*.png')))
                    frames = np.array(frames)
                    frame_sequences, pred_sequences = self.get_sequences(frames, self.seq_length, self.sampling)
                    for frames, preds in zip(frame_sequences, pred_sequences):
                        db.append(
                            {
                                'sequence': frames,
                                'preds': preds,
                                'subj_id': subj_id,
                            }
                        )

        return db

    def get_sequences(self, frames, length, sampling):
        frame_sequences = []
        pred_sequences = []
        indices = np.arange(0, length * sampling, sampling)
        for i in range((len(frames) - length)):
            try:
                frame_sequence = frames[indices + i]
            except IndexError:
                break
            has_skip = False
            frame_nums = [int(os.path.split(f)[1].split('_')[1].split('.')[0]) for f in frame_sequence]
            for j in range(len(frame_nums) - 1):
                if frame_nums[j] + sampling != frame_nums[j + 1]:
                    has_skip = True
            if not has_skip:
                frame_sequences.append(frame_sequence)
                split = frame_sequence[0].split('/')
                subj = split[-2]
                frame_ids = [f.split('/')[-1] for f in frame_sequence]
                pred_sequence = [self.predictions[os.path.join(self.img_path, subj, id)][self.indices]
                                 for id in frame_ids]
                pred_sequences.append(pred_sequence)

        return frame_sequences, pred_sequences

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        item = copy.deepcopy(self.db[idx])
        frames = [cv2.imread(f,  cv2.IMREAD_COLOR) for f in item['sequence']]
        frames_list = self.image_dict[item['subj_id']]
        num = int(os.path.splitext(os.path.split(item['sequence'][0])[1])[0].split('_')[1])
        excluded_indices = list(range(num-300, num+300))
        frames_list = [f for index, f in enumerate(frames_list) if index not in excluded_indices]

        second_image = random.choice(frames_list)
        second_image = cv2.imread(second_image, cv2.IMREAD_COLOR)
        preds = item['preds']

        subj = item['subj_id']

        c = self.center_dict[subj]['center']
        scale = self.center_dict[subj]['scale']
        s = np.asarray([scale, scale])

        trans = get_affine_transform(c, s, 0, self.image_size)
        tr = [cv2.warpAffine(f, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
              for f in frames]

        second_image = cv2.warpAffine(second_image, trans, (int(self.image_size[0]), int(self.image_size[1])),
                                      flags=cv2.INTER_LINEAR)

        if self.transform:
            tr = [self.transform(f) for f in tr]
            preds = [torch.from_numpy(p) for p in preds]

            tr = torch.stack(tr, dim=-1)
            preds = torch.stack(preds, dim=-1)

            second_image = self.transform(second_image)

        return tr, second_image, preds

    def get_img_dict(self):
        folders = glob(os.path.join(self.img_path, '*/'))
        img_dict = {}
        for folder in folders:
            subj_id = folder.split('/')[-2]
            sub_list = []
            frames = natsorted(glob(os.path.join(folder, '*.png')))
            for frame in frames:
                sub_list.append(frame)
            img_dict[subj_id] = sub_list

        return img_dict


class SequentialTrackerDataset(Dataset):
    def __init__(self, path, length, sample_rate):
        self.path = path
        self.sample_rate = sample_rate
        self.length = length
        self.indices = np.array([15, 14, 10, 6, 3, 0, 11, 7, 4, 1, 12, 8, 5, 2, 13, 9])
        self.keypoints = ['hips', 'shoulders', 'knees', 'elbows', 'ankles', 'wrists', 'feet', 'hands']
        self.sequences = self.get_sequences()

    def get_sequences(self):
        spatial_indices = np.array([1, 2, 0])
        sequences = []
        files = glob(os.path.join(self.path, 'tracker', '*.mat'))
        for file in files:
            # subj = os.path.splitext(os.path.split(file)[-1])[0]
            data, timestamps = load_matfile(file)
            # only xyz
            data = data[:, 0:3] #(length,3,16)
            data = data.transpose((0, 2, 1)) #(l,16,3)
            # tracker data is 120 Hz, camera is 30 Hz, so factor is 4
            tracker_data = data[::4*self.sample_rate]
            for i in range(len(tracker_data)-self.length):
                sequence = tracker_data[np.arange(0, self.length)+i]
                # change keypoint order and spatial orientation
                sequence = sequence[:, self.indices]
                sequence = sequence[..., spatial_indices]
                sequence = self.align(sequence)
                first = sequence[0]
                pelvis = first[0] + (first[1] - first[0]) / 2
                sequence -= pelvis
                sequence = self.normalize(sequence)
                sequences.append(sequence)
        return sequences

    def normalize(self, points):
        first = points[0]
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
        points = sequence[0]
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

        # hip
        rot_axis2 = np.array([1, 0])
        angle = self.get_angle(hip_line[:2], rot_axis2)
        cross_prod = np.cross(hip_line[0:2], rot_axis2)
        cross_prod /= np.linalg.norm(cross_prod)
        R2 = R.from_rotvec(angle * np.array([0, 0, cross_prod]))

        for i in range(len(sequence)):
            sequence[i] = R2.apply(R1.apply(sequence[i]))

        return sequence

    def __getitem__(self, idx):
        sequence = np.array(self.sequences[idx])
        s1 = np.copy(sequence)
        s1[..., 1] = sequence[..., 2]
        s1[..., 2] = sequence[..., 1]
        # 180 around y axis
        rot = R.from_rotvec(np.array([0., 3.14159, 0.]))
        for i in range(len(s1)):
            s1[i] = rot.apply(s1[i])
        s1 = [torch.from_numpy(p) for p in s1]
        sequence = torch.stack(s1, dim=-1)

        return sequence

    def __len__(self):
        return len(self.sequences)