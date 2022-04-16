# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: gen_edge.py
@time: 2021/5/10 15:36
"""

import cv2
import os

def Edge_Extract(root):
    img_root = os.path.join(root,'GT')
    edge_root = os.path.join(root,'Edge')

    if not os.path.exists(edge_root):
        os.mkdir(edge_root)

    file_names = os.listdir(img_root)
    img_name = []

    for name in file_names:
        print(f'Generate Edge Image {name} successful!')
        if not name.endswith('.png'):
            assert "This file %s is not PNG"%(name)
        img_name.append(os.path.join(img_root,name[:-4]+'.png'))

    index = 0
    for image in img_name:
        img = cv2.imread(image,0)
        cv2.imwrite(edge_root+'/'+file_names[index],cv2.Canny(img,30,100))
        index += 1
    return 0


if __name__ == '__main__':
    root = r'F:\tanyacheng\tanyacheng\RGB-T SOD\VT5000\VT5000_clear\Train'
    Edge_Extract(root)
