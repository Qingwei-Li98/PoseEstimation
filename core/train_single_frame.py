import sys
import os
import torchvision.transforms as transforms
curPath = os.path.abspath(os.path.dirname("sequential_model.py"))
print(curPath)
rootPath = os.path.split(curPath)[0]
print(rootPath)
sys.path.append(rootPath)
sys.path.append(curPath)
# import utils.datasets_2 as ds
import utils.datasets_processed as ds
import warnings
warnings.filterwarnings("ignore")
import torch 
import torch.nn as nn
import trainer
import train_test
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

# activities = ['directions']
# train_actors=['S%d' % i for i in [1]]

activities = ['directions', 'discussion','eating','purchases', 'greeting','phoning', 'posing','purchases',
                      'waiting', 'walking','walktogether','sitting','walkdog','sittingdown','smoking','photo']
# activities = ['directions', 'discussion', 'greeting', 'posing',
#                       'waiting', 'walking']
actors = ['S%d' % i for i in [1, 5, 6, 7, 8]]
trans=transforms.Compose([
        transforms.ToTensor()
    ])
# trans=transforms.Compose([
#         transforms.Resize((256,256)),
#         transforms.ToTensor()
#     ])
# root='/vol/biodata/data/h36m/training'
# pose_root='/vol/biodata/data/h36m/h36m_poses/d2'
# pose3d_root='/vol/biodata/data/h36m/h36m_poses/d3/'
# path='./sequence_ds2/sequences.tar'
# data=ds.Human36M(root,pose_root,activities=activities, actors=actors,seq_path=path,random_seed=2,transforms=trans, is_train=True, subsampled_size=None)
# pose_data=ds.Human36MPose(pose3d_root,data.get_sequences())

root='/vol/biodata/data/h36m_processed/training/'
pose_root='/vol/biodata/data/h36m_poses/d2/'
pose3d_root='/vol/biodata/data/h36m_poses/d3/'
path='/vol/bitbucket/ql1220/IndividualProject/poseestimate/core/sequences/sequences.tar'
# path_less='/vol/bitbucket/ql1220/IndividualProject/poseestimate/core/sequence_less'
data=ds.Human36M(root,pose_root,activities,actors,path=path)
# data.save_seq(path)
pose_data=ds.Human36MPose(pose3d_root,data.get_sequences())

torch.cuda.empty_cache()

# logdir='/vol/bitbucket/ql1220/out/log0731'
# checkdir='/vol/bitbucket/ql1220/out/check0731'
logdir='/vol/bitbucket/ql1220/out/log0803_vgg'
checkdir='/vol/bitbucket/ql1220/out/check0803_vgg'
# loaddir='/vol/bitbucket/ql1220/out/check0725/checkpoint_2.0.tar'
model=trainer.ForwardKinematics(data,pose_data)
loadpath='/vol/bitbucket/ql1220/out/check0802_vgg'
model.load_checkpoint(loadpath+'/checkpoint_epoch1.tar')
# model=train_test.ForwardKinematics(data,pose_data)
# model.load_checkpoint(loaddir)

model.train(30,2,logdir,checkdir)
