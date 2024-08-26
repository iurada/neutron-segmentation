from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset as torchDataset
import os
from abc import ABCMeta
from dataclasses import dataclass
from typing import Tuple, Optional
from utils.ext_transform import ExtTransforms, ExtToTensor
from datasets.cityscapes import Cityscapes
class BaseGTALabels(metaclass=ABCMeta):
    pass

@dataclass
class GTA5Label:
    ID: int
    color: Tuple[int, int, int]


class GTA5Labels_TaskCV2017(BaseGTALabels):
    road = GTA5Label(ID=0, color=(128, 64, 128))
    sidewalk = GTA5Label(ID=1, color=(244, 35, 232))
    building = GTA5Label(ID=2, color=(70, 70, 70))
    wall = GTA5Label(ID=3, color=(102, 102, 156))
    fence = GTA5Label(ID=4, color=(190, 153, 153))
    pole = GTA5Label(ID=5, color=(153, 153, 153))
    light = GTA5Label(ID=6, color=(250, 170, 30))
    sign = GTA5Label(ID=7, color=(220, 220, 0))
    vegetation = GTA5Label(ID=8, color=(107, 142, 35))
    terrain = GTA5Label(ID=9, color=(152, 251, 152))
    sky = GTA5Label(ID=10, color=(70, 130, 180))
    person = GTA5Label(ID=11, color=(220, 20, 60))
    rider = GTA5Label(ID=12, color=(255, 0, 0))
    car = GTA5Label(ID=13, color=(0, 0, 142))
    truck = GTA5Label(ID=14, color=(0, 0, 70))
    bus = GTA5Label(ID=15, color=(0, 60, 100))
    train = GTA5Label(ID=16, color=(0, 80, 100))
    motocycle = GTA5Label(ID=17, color=(0, 0, 230))
    bicycle = GTA5Label(ID=18, color=(119, 11, 32))

    list_ = [
        road,
        sidewalk,
        building,
        wall,
        fence,
        pole,
        light,
        sign,
        vegetation,
        terrain,
        sky,
        person,
        rider,
        car,
        truck,
        bus,
        train,
        motocycle,
        bicycle,
    ]

    @property
    def support_id_list(self):
        ret = [label.ID for label in self.list_]
        return ret



class GTA5(torchDataset):
    label_map = GTA5Labels_TaskCV2017()
    id_to_train_id = np.array([c.train_id for c in Cityscapes.classes])
    id_to_train_id = np.append(id_to_train_id, 255)
    
    class PathPair_ImgAndLabel:
        IMG_DIR_NAME = "images"
        LBL_DIR_NAME = "labels"
        SUFFIX = ".png"

        def __init__(self, root, split, labels_source="train_ids"):
            self.root = root
            self.labels_source = labels_source
            self.split = split
            self.img_paths = self.create_imgpath_list()
            self.lbl_paths = self.create_lblpath_list()
            
        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx: int):
            img_path = self.img_paths[idx]
            lbl_path = self.lbl_paths[idx]
            return img_path, lbl_path

        def create_imgpath_list(self):
            #img_dir = os.path.join(self.root , self.IMG_DIR_NAME,self.split)
            with open(os.path.join(self.root, f'{self.split}.txt'), 'r') as f:
                paths = f.readlines()
            img_path = [os.path.join(self.root, path.strip().split()[0][1:]) for path in paths]
            return img_path

        def create_lblpath_list(self):
            #lbl_dir = os.path.join(self.root,self.LBL_DIR_NAME,self.split)
            #if self.labels_source == "cityscapes":
            #    lbl_path = [os.path.join(lbl_dir,path) for path in os.listdir(lbl_dir) if path.endswith(self.SUFFIX) and path.__contains__("_labelTrainIds")]
            #elif self.labels_source == "GTA5":
            #    lbl_path = [os.path.join(lbl_dir,path) for path in os.listdir(lbl_dir) if (path.endswith(self.SUFFIX) and not path.__contains__("_labelTrainIds.png"))]
            with open(os.path.join(self.root, f'{self.split}.txt'), 'r') as f:
                paths = f.readlines()
            lbl_path = [os.path.join(self.root, path.strip().split()[1][1:]) for path in paths]
            return lbl_path

    def __init__(self, 
                 root: Path,
                 labels_source: str = "GTA5", # "cityscapes" or "GTA5"
                 transforms:Optional[ExtTransforms]=None,
                 split="train"):
        """

        :param root: (Path)
            this is the directory path for GTA5 data
        """
        self.root = root #os.path.join(root , 'GTA5')
        self.labels_source = labels_source
        self.transforms = transforms
        self.paths = self.PathPair_ImgAndLabel(root=self.root,split=split,labels_source=labels_source)
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx, isPath=False):
        img_path, lbl_path = self.paths[idx]
        if isPath:
            return img_path, lbl_path
        img = self.read_img(img_path)
        lbl = self.read_img(lbl_path)
        
        if self.labels_source == "GTA5":
            lbl = Image.fromarray(np.array(self.map_to_cityscapes(lbl),dtype='uint8')) 
            #if not os.path.exists(lbl_path.split('.png')[0] + "_labelTrainIds.png"):
            #    lbl.convert('L').save(lbl_path.split('.png')[0] + "_labelTrainIds.png")
        
        if self.transforms is not None:
            img, lbl = self.transforms(img, lbl)
        else:
            img = ExtToTensor()(img)
            lbl = ExtToTensor()(lbl)
        return img, lbl

    @staticmethod
    def read_img(path):
        img = Image.open(str(path))
        #img = np.array(img)
        return img

    @classmethod
    def decode(cls, lbl):
        return cls._decode(lbl, label_map=cls.label_map.list_)

    @staticmethod
    def _decode(lbl, label_map):
        # remap_lbl = lbl[np.where(np.isin(lbl, cls.label_map.support_id_list), lbl, 0)]
        color_lbl = np.zeros((*lbl.shape, 3))
        for label in label_map:
            color_lbl[lbl == label.ID] = label.color
        return color_lbl
    
    def map_to_cityscapes(self, lbl):
        return self.id_to_train_id[np.array(lbl)]