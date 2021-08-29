import sys
import os
from utils import datasets_2 as ds
import torchvision.transforms as transforms
curPath = os.path.abspath(os.path.dirname("sequential_model.py"))
print(curPath)
rootPath = os.path.split(curPath)[0]
print(rootPath)
sys.path.append(rootPath)
sys.path.append(curPath)
import warnings
warnings.filterwarnings("ignore")
import torch 
import torch.nn as nn
import trainer
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

# activities = ['directions']
train_actors=['S%d' % i for i in [1]]
# activities = ['directions', 'discussion', 'eating', 'greeting','phoning','photo','posing','purchases'
#                   'waiting', 'walking','walkdog', 'walktogether', 'sitting','sittingdown','smoking']
activities = ['directions', 'discussion','eating','purchases', 'greeting', 'posing',
                      'waiting', 'walking','walktogether','sitting','smoking','photo']
# train_actors = ['S%d' % i for i in [1, 5, 6, 7, 8, 9]]
trans=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
root='/vol/biodata/data/h36m/training'
pose_root='/vol/biodata/data/h36m/h36m_poses/d2'
data=ds.Human36M2(root,pose_root,activities=activities, actors=train_actors,random_seed=2,transforms=trans, is_train=True, subsampled_size=None)

pose3d_root='/vol/biodata/data/h36m/h36m_poses/d3/'
pose_data=ds.Human36MPose2(pose3d_root,data.get_sequences())
torch.cuda.empty_cache()

logdir='/vol/bitbucket/ql1220/out/log_test30726'
checkdir='/vol/bitbucket/ql1220/out/check_test30726'
# loaddir='/vol/bitbucket/ql1220/out/check0725/checkpoint_2.0.tar'
model=trainer.ForwardKinematics(data,pose_data)
# model.load_checkpoint(loaddir)

model.train(35,3,logdir,checkdir)
