import sys
import os
curPath = os.path.abspath(os.path.dirname("sequential_model.py"))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import torch
from utils import datasets as ds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import cv2
import sequential_model

if __name__ == '__main__':
    print('Successful ')