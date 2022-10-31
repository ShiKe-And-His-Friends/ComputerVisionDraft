import os.path
import numpy as np
from pyquaternion import Quaternion
from data_classes import Box

from torch.utils.data import Dataset

class kittiDataset():
    def __init__(self ,path):
        self.KITTI_Folder = path
        self.KITTI_velo = os.path.join(self.KITTI_Folder ,"velodyne")
        self.KITTI_label = os.path.join(self.KITTI_Folder ,"label_02")

    def getSceneID(self ,split):
        if "TRAIN" in split.upper(): # Training SET
            if "TINY" in split.upper():
                sceneID = [0]
            else:
                sceneID = list(range(0 ,17))
        elif "VALID" in split.upper():
            if "TINY" in split.upper():
                sceneID = [18]
            else:
                sceneID = list(range(17 ,19))
        elif "TEST" in split.upper():
            if "TINY" in split.upper():
                sceneID = [19]
            else:
                sceneID = list(range(19,21))
        else: # Full Dataset
            sceneID = list(range(21))
        return sceneID

    def getBBandPC(self ,anno):
        calib_path = os.path.join(self.KITTI_Folder ,'calib' ,anno['scene']+".txt")
        calib = self.read_calib_file(calib_path)
        transf_mat = np.vstack((calib["Tr_velo_cam"] ,np.array([0 ,0 ,0 ,1])))
        PC ,box = self.getPCandBBfromPandas(anno ,transf_mat)

    def getPCandBBfromPandas(self ,box ,calib):
        center = [box["x"] ,box["y"] - box["height"] / 2 ,box["z"]]
        size = [box["width"] ,box["length"] ,box["height"]]
        orientation = \
            Quaternion(
                axis = [0 ,1 ,0],
                radians = box["rotation_y"]) * \
            Quaternion(
                axis = [1 ,0 ,0],
                radians = np.pi / 2
            )
        BB = Box(center ,size ,orientation)

    def read_calib_file(self ,filepath):
        """ Read in a calibration file and parse into a dictionary """
        data = {}
        with open(filepath ,"r") as f:
            for line in f.readlines():
                values = line.split()
                # the only non-float values in these files are dates , which
                # we don't care about anyway
                try:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]
                    ).reshape(3,4)
                except ValueError:
                    pass

class SiameseDataset(Dataset):
    def __init__(self,
                 input_size,
                 path,
                 split,
                 category_name = "Car",
                 regress = "GAUSSIAN",
                 offset_BB = 0,
                 scale_BB = 1.0):
        self.dataset = kittiDataset(path = path)
        self.input_size = input_size
        self.split = split
        self.sceneID = self.dataset.getSceneID(split = split)
        self.geyBBandPC = self.dataset.getBBandPC

class SiameseTrain():