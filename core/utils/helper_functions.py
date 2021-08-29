import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
from tqdm import tqdm
import os.path
import tarfile


def load_matfile(file):
    file = loadmat(file)
    data = None
    for key in ['save_data', 'data', 'corrected_data']:
        try:
            data = file[key]
        except KeyError:
            pass
    if data is None:
        print('could not load data, check for a missing key!')
        raise
    timestamps = file['time_data'][0]

    # filter out nan values
    indices = np.unique(np.nonzero(~np.isnan(data))[0])
    data = data[indices]
    timestamps = timestamps[indices]

    return data, timestamps


def plot3d(infant_pose, path):
    """input is tensor of size [b, num_points, 3]"""
    infant_pose = infant_pose[0].detach().cpu().numpy()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(infant_pose[:, 0], infant_pose[:, 1], infant_pose[:, 2])

    indices1 = np.asarray([12, 8, 4, 0, 1, 5, 9, 13])
    indices2 = np.asarray([15, 11, 7, 3, 2, 6, 10, 14])
    indices3 = np.asarray([0, 2])
    indices4 = np.asarray([1, 3])

    data1 = infant_pose[indices1, :]
    data2 = infant_pose[indices2, :]
    data3 = infant_pose[indices3, :]
    data4 = infant_pose[indices4, :]

    ax.plot(data1[:, 0], data1[:, 1], data1[:, 2],'y') # down
    ax.plot(data2[:, 0], data2[:, 1], data2[:, 2],'g') # up
    ax.plot(data3[:, 0], data3[:, 1], data3[:, 2],'r') # left
    ax.plot(data4[:, 0], data4[:, 1], data4[:, 2],'b') # right

    ax.scatter([0], [0], [2.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(path)
    plt.close()
    
def get_file_num(path):
    file_num = sum([os.path.isfile(os.path.join(path, listx)) for listx in os.listdir(path)])
    return file_num

def plot2d(pose, path):
    """input is tensor of size [b, num_points, 2]"""
    points=pose[0].detach().cpu().numpy()
    indices1 = np.asarray([12, 8, 4, 0, 1, 5, 9, 13])
    indices2 = np.asarray([15, 11, 7, 3, 2, 6, 10, 14])
    indices3 = np.asarray([0, 2])
    indices4 = np.asarray([1, 3])
    pose1=points[indices1,:]
    pose2=points[indices2,:]
    pose3=points[indices3,:]
    pose4=points[indices4,:]
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.scatter(points[:, 0], points[:, 1])
    ax.plot(pose1[:,0],pose1[:,1],'y') #down
    ax.plot(pose2[:,0],pose2[:,1],'g') #up
    ax.plot(pose3[:,0],pose3[:,1],'r') #right
    ax.plot(pose4[:,0],pose4[:,1],'b') #left
    plt.gca().invert_yaxis()
    plt.savefig(path)
    plt.close()
    

def save_3dseq(infant_poses, path):
    """input is tensor of size [b, num_points, 3]"""
    infant_poses = infant_poses.detach().cpu().numpy()
    path=os.path.join(path)
    if not os.path.exists(path):
            os.makedirs(path)
    for i in range(len(infant_poses)):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(projection='3d')
        infant_pose=infant_poses[i]
        ax.scatter(infant_pose[:, 0], infant_pose[:, 1], infant_pose[:, 2])

        indices1 = np.asarray([12, 8, 4, 0, 1, 5, 9, 13])
        indices2 = np.asarray([15, 11, 7, 3, 2, 6, 10, 14])
        indices3 = np.asarray([0, 2])
        indices4 = np.asarray([1, 3])

        data1 = infant_pose[indices1, :]
        data2 = infant_pose[indices2, :]
        data3 = infant_pose[indices3, :]
        data4 = infant_pose[indices4, :]

        ax.plot(data1[:, 0], data1[:, 1], data1[:, 2])
        ax.plot(data2[:, 0], data2[:, 1], data2[:, 2])
        ax.plot(data3[:, 0], data3[:, 1], data3[:, 2])
        ax.plot(data4[:, 0], data4[:, 1], data4[:, 2])
        ax.view_init(45,-60)
        ax.scatter([0], [0], [2.5])
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.savefig(os.path.join(path,'frame'+str(i)))
    tar = tarfile.open(os.path.join(path,'3dpose.tar'), 'w')
    for files in os.listdir(path):   
        tar.add(os.path.join(path,files))
    tar.close()
    plt.close()
    
def save_2dseq(frames,pose, path):
    """input is tensor of size [b, num_points, 2]"""
    points=pose.detach().cpu().numpy()
    if not os.path.exists(path):
            os.makedirs(path)
    indices1 = np.asarray([12, 8, 4, 0, 1, 5, 9, 13])
    indices2 = np.asarray([15, 11, 7, 3, 2, 6, 10, 14])
    indices3 = np.asarray([0, 2])
    indices4 = np.asarray([1, 3])
    pose1=points[:,indices1,:]
    pose2=points[:,indices2,:]
    pose3=points[:,indices3,:]
    pose4=points[:,indices4,:]
    for i in range(len(pose)):
        point=points[i]
        frame=frames[i]
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot()
        plt.imshow(frame.permute(1,2,0))
        ax.scatter(point[:, 0], point[:, 1],marker='.')
        ax.plot(pose1[i,:,0],pose1[i,:,1])
        ax.plot(pose2[i,:,0],pose2[i,:,1])
        ax.plot(pose3[i,:,0],pose3[i,:,1])
        ax.plot(pose4[i,:,0],pose4[i,:,1])
        ax.set_xlim([0,256])
        ax.set_ylim([0,256])
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(path,'frame'+str(i)))
    tar = tarfile.open(os.path.join(path,'2dpose.tar'), 'w')
    for files in os.listdir(path):   
        tar.add(os.path.join(path,files))
    tar.close()
    plt.close()

def save_images(imgs,path):
    """
        imgs: tensor with size of (batch,c,h,w)
        path: path to save
    """
    imgs=imgs.permute(0,2,3,1).detach().cpu().numpy()
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(len(imgs)):
        img=imgs[i]
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.imshow(img)
        plt.savefig(os.path.join(path,'frame'+str(i)))
    tar = tarfile.open(os.path.join(path,'frames.tar'), 'w')
    for files in os.listdir(path):   
        tar.add(os.path.join(path,files))
    tar.close()
    plt.close()
    

def tar_file(src,target):
    tar = tarfile.open(target, 'w')
    for files in tqdm(os.listdir(src)):
        tar.add(os.path.join(src,files))
    tar.close()

    
