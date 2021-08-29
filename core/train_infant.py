import warnings
warnings.filterwarnings("ignore")
import os
import sys
import numpy as np
curPath = os.path.abspath(os.path.dirname("sequential_model.py"))
print(curPath)
rootPath = os.path.split(curPath)[0]
print(rootPath)
sys.path.append(rootPath)
sys.path.append(curPath)
import utils.infant_dataset as infant_ds
import torch.nn as nn
import trainer

root='/vol/biodata/data/infant_data/frames_cut'
pose_root='/vol/biodata/data/infant_data/predictions'
# subjects=['EMT4', 'EMT7', 'EMT38', 'EMT36', 'EMT31', 'EMT43',
#             'EMT5', 'EMT9', 'EMT47', 'EMT45', 'EMT29', 'EMT42',
#             'EMT23', 'EMT41', 'EMT37', 'EMT48', 'EMT44', 'EMT46',
#             'EMT20', 'EMT11', 'EMT30', 'EMT39', 'EMT35', 'EMT14',
#           'EMT60', 'EMT84', 'EMT79', 'EMT90', 'EMT83','EMT70']
subjects=[ 'EMT38', 'EMT31', 'EMT43', 'EMT45',  'EMT42',
          'EMT41', 'EMT44', 'EMT39', 'EMT14', 'EMT84',
          'EMT60','EMT58','EMT52', 'EMT73','EMT97',
          'EMT55','EMT67','EMT89','EMT63','EMT87',
         'EMT51','EMT90','EMT84','EMT60','EMT79',
          'EMT83','EMT70']
subjects_test=['EMT29','EMT30','EMT35','EMT37']
save_path='./infant_sequences_processed'
train_subjects=subjects[:-1]
val_subjects=subjects[-1:]
infant_train_data=infant_ds.InfantDataSet(root,pose_root,train_subjects)
infant_val_data=infant_ds.InfantDataSet(root,pose_root,val_subjects)
infant_train_data.save_seq(save_path)
pose3d_root='/vol/biodata/data/infant_data/tracker/'
# load_path='./infant_3dposes/pose3d_sequences.tar'
model_path='/vol/bitbucket/ql1220/out/check0802_vgg/checkpoint_epoch1.tar'
# model_path='/vol/bitbucket/ql1220/out/infant_check_pretrained/checkpoint_epoch14.tar'
infant_pose=infant_ds.Infant3DPose(pose3d_root)
logdir='/vol/bitbucket/ql1220/out/infant_log_finetune_pretrained'
checkdir='/vol/bitbucket/ql1220/out/infant_check_finetune_pretrained'
model=trainer.ForwardKinematics(infant_train_data,infant_val_data,infant_pose)
model.load_pretrained_checkpoint(model_path)
model.train(30,15,logdir,checkdir)