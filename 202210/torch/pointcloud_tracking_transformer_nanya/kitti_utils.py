import copy

import torch
from pyquaternion import Quaternion
from data_classes import PointCloud
import numpy as np

def getOffsetBB(box ,offset ,training = True):
    rot_quat = Quaternion(matrix=box.rotation_matrix)
    trans = np.array(box.center)

    new_box = copy.deepcopy(box)
    new_box.translate(-trans)
    new_box.rotate(rot_quat.inverse)

    # REMOVE TRANSFORM
    if len(offset) == 3:
        assert False
        #new_box.rotate(
        #    Quaternion(axis=[0, 0, 1], angle=offset[2] * np.pi / 180)
        #)
    elif len(offset) == 4:
        new_box.rotate(
            Quaternion(axis=[0,0,1] ,angle=offset[3] * np.pi / 180)
        )
    if training and np.abs(offset[0]) > min(new_box.wlh[1] ,3):
        offset[0] = np.random.uniform(0 ,min(new_box.wlh[1] ,3)) * np.sign(offset[0])
    if training and np.abs(offset[1]) > new_box.wlh[0]:
        offset[1] = np.random.uniform(0 ,new_box.wlh[0]) * np.sign(offset[1])
    if training and np.abs(offset[2]) > new_box.wlh[2]:
        offset[2] = np.random.uniform(0 ,new_box.wlh[2]) * np.sign(offset[2])
    new_box.translate(np.array([offset[0] ,offset[1],offset[2]]))

    #APPLY PREVIOUS TRANSFORMATION
    new_box.rotate(rot_quat)
    new_box.translate(trans)
    return new_box

def regularizePCwithlabel(PC ,label ,reg ,input_size ,istrain=True ,keep_first_half=False):
    PC = np.array(PC.points ,dtype=np.float32)
    if np.shape(PC)[1] > 2:
        if PC.shape[0] > 3:
            PC = PC[0:3 , :]
        if PC.shape[1] != input_size:
            if not istrain:
                np.random.seed(1)
            if PC.shape[1] >= input_size:
                idx = np.arange(PC.shape[1])
                np.random.shuffle(idx)
                idx = idx[:input_size]

                PC = PC[: ,idx]
                label = label[idx]
            else:
                new_pc_idx = np.random.randint(low=0 ,high=PC.shape[1] ,size=input_size -PC.shape[1])
                PC = np.concatenate([PC ,PC[: ,new_pc_idx]] ,axis = -1)
                label = np.concatenate([label ,label[new_pc_idx]] ,axis=0)
        PC =PC.reshape((3,input_size)).T
        label = label[0 : input_size //2 ]
        reg = np.tile(reg ,[np.size(label) ,1])
    else:
        PC = np.zeros((3,input_size)).T
        label = np.zeros(128)
        reg = np.tile(reg ,[np.size(label) ,1])

    return torch.from_numpy(PC).float() ,\
            torch.from_numpy(label).float(),\
            torch.from_numpy(reg).float()

def getModel(PCs ,boxes ,offset = 0 ,scale =1.0 ,normalize =False):
    if len(PCs) == 0:
        return PointCloud(np.ones((3,0)))
    points = np.ones((PCs[0].points.shape[0] ,0))

    for PC , box in zip(PCs ,boxes):
        cropped_PC = cropAndCenterPC(
            PC ,box ,offset=offset ,
            scale = scale ,normalize= normalize
        )
        # try:
        if cropped_PC.points.shape[1] > 0:
            points = np.concatenate([points ,cropped_PC.points] ,axis=1)
    PC = PointCloud(points)
    return PC

def cropPC(PC ,box ,offset=0 ,scale=1.0):
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners() ,1) + offset
    mini = np.min(box_tmp.corners() ,1) - offset

    x_filt_max = PC.points[0 ,:] < maxi[0]
    x_filt_min = PC.points[0 ,:] > mini[0]
    y_flit_max = PC.points[1 ,:] < maxi[1]
    y_flit_min = PC.points[1 ,:] > mini[1]
    z_flit_max = PC.points[2 ,:] < maxi[2]
    z_flit_min = PC.points[2 ,:] > mini[2]

    close = np.logical_and(x_filt_min ,x_filt_max)
    close = np.logical_and(close ,y_flit_min)
    close = np.logical_and(close ,y_flit_max)
    close = np.logical_and(close ,z_flit_min)
    close = np.logical_and(close ,z_flit_max)

    new_PC = PointCloud(PC.points[: ,close])
    return new_PC

def getlabelPC(PC ,box ,offset=0 ,scale=1.0):
    """
    1. align :move to box center
    2. scale
    3. offset
    4. pt inside box = 1
    """
    box_tmp = copy.deepcopy(box)
    new_PC = PointCloud(PC.points.copy())
    rot_mat = np.transpose(box_tmp.rotation_matrix)
    trans = -box_tmp.center

    #align data
    new_PC.translate(trans)
    box_tmp.translate(trans)
    new_PC.rotate(rot_mat)
    box_tmp.rotation(Quaternion(matrix=(rot_mat)))

    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners() ,1) + offset
    mini = np.min(box_tmp.corners() ,1) - offset

    x_filt_max = new_PC.points[0, :] < maxi[0]
    x_filt_min = new_PC.points[0, :] > mini[0]
    y_flit_max = new_PC.points[1, :] < maxi[1]
    y_flit_min = new_PC.points[1, :] > mini[1]
    z_flit_max = new_PC.points[2, :] < maxi[2]
    z_flit_min = new_PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_flit_min)
    close = np.logical_and(close, y_flit_max)
    close = np.logical_and(close, z_flit_min)
    close = np.logical_and(close, z_flit_max)

    new_label = np.zeros(new_PC.points.shape[1])
    new_label[close] = 1
    return new_label

def cropPCwithlabel(PC ,box ,label ,offset=0 ,scale=1.0):
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners() ,1) + offset
    mini = np.max(box_tmp.corners() ,1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_flit_max = PC.points[1, :] < maxi[1]
    y_flit_min = PC.points[1, :] > mini[1]
    z_flit_max = PC.points[2, :] < maxi[2]
    z_flit_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_flit_min)
    close = np.logical_and(close, y_flit_max)
    close = np.logical_and(close, z_flit_min)
    close = np.logical_and(close, z_flit_max)
    new_PC = PointCloud(PC.points[:, close])
    new_label = label[close]
    return new_PC ,new_label

def cropAndCenterPC(PC,box ,offset= 0 ,scale=1.0 ,normalize = False):
    new_PC = cropPC(PC ,box ,offset = 2 * offset ,scale = 4 * scale)
    new_box = copy.deepcopy(box)
    rot_mat = np.transpose(new_box.rotate_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC = cropPC(new_PC ,new_box ,offset = offset ,scale=scale)
    if normalize:
        new_PC.normalize(box.wlh)
    return new_PC

def cropAndCenterPC_label(PC ,sample_box ,gt_box ,sample_offsets,
                          offset = 0 ,scale=1.0 ,normalize = False):
    """
    1.get pc inside 4 * sample_box
    2.label pc inside gt_box
    3.move to sample_box
    4.crop pc inside sample_box
    5.gt_box as label_reg
    """
    new_PC = cropPC(PC, sample_box ,offset=2 * offset ,scale = 4 * scale)
    new_box = copy.deepcopy(sample_box)

    new_label = getlabelPC(new_PC , gt_box ,offset = offset ,scale=1.0)
    new_box_gt = copy.deepcopy(gt_box)
    rot_mat = np.transpose(new_box.rotate_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)
    new_PC.rotate(rot_mat)
    new_box.rotate(Quaternion(matrix=rot_mat))

    new_box_gt.translate(trans)
    new_box_gt.rotate(Quaternion(matrix=rot_mat))

    # crop around box
    new_PC ,new_label = cropPCwithlabel(
        new_PC,new_box,new_label,
        offset = offset+2.0 ,scale=1 *scale
    )
    label_reg = [
        new_box_gt.center[0],
        new_box_gt.center[1],
        new_box_gt.center[2],
        -sample_offsets[-1]
    ]
    label_reg = np.array(label_reg)

    if normalize:
        new_PC.normalize(sample_box.wlh)
    return new_PC,new_label ,label_reg