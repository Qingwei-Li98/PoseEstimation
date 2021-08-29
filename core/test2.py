import sys
import os
curPath = os.path.abspath(os.path.dirname("sequential_model.py"))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)
os.environ["CDF_LIB"] = '/vol/bitbucket/ql1220/envs/pose_env/lib/cdf38_0-dist/lib'
import torch
from utils import datasets as ds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import cv2
import sequential_model
import warnings
warnings.filterwarnings("ignore")

def main():
    activities = ['directions', 'discussion', 'eating', 'greeting','phoning','photo','posing','purchases'
                      'waiting', 'walking','walkdog', 'walktogether', 'sitting','sittingdown','smoking']
#     activities = ['directions', 'sitting']
    train_actors = ['S%d' % i for i in [1, 5, 6, 7, 8,9]]
#     activities = ['directions']
#     train_actors=['S%d' % i for i in [1]]
    trans=transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    root='/vol/biodata/data/h36m/training'
    pose_root='/vol/biodata/data/h36m/h36m_poses/d2'
    pose3d_root='/vol/biodata/data/h36m/h36m_poses/d3'
    h=ds.Human36M(root,pose_root,activities=activities, actors=train_actors,sequence_len=20,random_seed=2,transforms=trans, is_train=True, subsampled_size=None)
    pose3d=ds.Human36MPose(pose3d_root,h.get_sequences())
    torch.cuda.empty_cache()
    model=sequential_model.ForwardKinematics(h,pose3d)
    logdir='/vol/bitbucket/ql1220/out/log_train2'
    checkdir='/vol/bitbucket/ql1220/out/check_train2'
    model.train(1, 1, logdir, checkdir)

if __name__ == '__main__':
    main()