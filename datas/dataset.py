# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: dataset.py
@time: 2021-05-04 22:22
"""

import torch
import numpy as np
import random
from torchvision.transforms import functional as F
from torch.utils import data
import cv2
from PIL import Image
import numbers
import os
import logging

img_size = 224

def load_image(path):
    if not os.path.exists(path):
        print(f'File {path} not exists')
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    img_size = tuple(in_.shape[:2])
    im = cv2.resize(im,(img_size,img_size))
    in__ = np.array(im, dtype=np.float32)
    in__ = in_.transpose((2, 0, 1))
    return in__, img_size

def load_mask(path):
    if not os.path.exists(path):
        print(f'File {path} not exists')
    im = Image.open(path).convert('L')
    im = im.resize((img_size, img_size), Image.ANTIALIAS)
    mask = np.array(im, dtype=np.float32)
    if len(mask.shape) == 3:
        mask = mask[:,:,0]
    mask = mask / 255.
    mask = mask[np.newaxis, ...]
    return mask

def cv_random_flip(img, label):
    flip_flag = random.randint(0,1)
    if flip_flag == 1:
        img = img[:, :, ::-1].copy()
        label = label[:, :, ::-1].copy()
    return img, label


class ImageDataTrain(data.Dataset):
    def __init__(self, data_root, data_list):
        # data_root 是图片的根目录
        # data_list 是图片名文件保存的目录
        print("Start load Train_Data")
        logging.info("Start load Train_Data")
        self.sal_root = data_root
        self.sal_source = data_list
        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]
        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        im_name = self.sal_list[item % self.sal_num].split()[0]
        gt_name = self.sal_list[item % self.sal_num].split()[1]

        sal_image, img_size = load_image(os.path.join(self.sal_root, im_name))
        sal_label = load_mask(os.path.join(self.sal_root, gt_name))
        sal_image, sal_label = cv_random_flip(sal_image, sal_label)

        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)

        sample = {'sal_image': sal_image, 'img_name': im_name, 'sal_label': sal_label, 'img_size':img_size}
        return sample

    def __len__(self):
        return self.sal_num

class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list):
        self.sal_root = data_root
        self.sal_source = data_list

        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]
        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        img_name = self.sal_list[item % self.sal_num].split()[0]
        image, image_size = load_image(os.path.join(self.sal_root,img_name))
        img = torch.Tensor(image)

        sample = {'image':img, 'image_name':img_name, 'image_size':image_size}
        return sample

def get_loader(config, mode='train', pin=False):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config['train_root_t'],config['train_list_t'])
        data_loader = data.DataLoader(
            dataset = dataset,
            batch_size = config['batch_size'],
            shuffle = shuffle,
            num_workers = config['num_thread'],
            pin_memory=pin
        )
    else:
        dataset = ImageDataTest(config['test_root'], config['test_list'])
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            shuffle=shuffle,
            num_workers=config['num_thread'],
            pin_memory=pin
        )
    return data_loader