# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 15:02:18 2021

@author: LBY
"""
from glob import glob

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2

def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)


def get_bbox(p0, p1):
    """
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    """
    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)

    return v, e


training_set = np.loadtxt('labels.csv', skiprows=1, dtype=str, delimiter=',')
ii = 0
for training_example in training_set:
    # print(training_example)
    path_tail = training_example[0]
    path_for_image = "train/trainval/" + path_tail + "_image.jpg"
    # path_for_xyz = "train/trainval/" + path_tail + "_image.jpg"
    path_for_proj = "train/trainval/" + path_tail + "_proj.bin"
    path_for_bbox = "train/trainval/" + path_tail + "_bbox.bin"
    # readImage = cv2.imread(path_for_image)
    readImage = plt.imread(path_for_image)
    rgbImage = cv2.cvtColor(readImage, cv2.COLOR_BGR2RGB)
    #cv2.imshow(f'output{ii}', rgbImage)
    #fig1 = plt.figure(1, figsize=(16, 9))
    #ax1 = fig1.add_subplot(1, 1, 1)
    #ax1.imshow(readImage)
    proj = np.fromfile(path_for_proj, dtype=np.float32)
    proj.resize([3, 4])
    ii = ii + 1
    print(ii)
    bbox = np.fromfile(path_for_bbox, dtype=np.float32)
#    try:
#        bbox = np.fromfile(path_for_bbox, dtype=np.float32)
#    except FileNotFoundError:
#        print('[*] bbox not found.')
#        bbox = np.array([], dtype=np.float32)
        
    bbox = bbox.reshape([-1, 11])


    colors = ['C{:d}'.format(i) for i in range(10)]
    for k, b in enumerate(bbox):
        # print(k)
        R = rot(b[0:3])
        t = b[3:6]
    
        sz = b[6:9]
        vert_3D, edges = get_bbox(-sz / 2, sz / 2)
        vert_3D = R @ vert_3D + t[:, np.newaxis]
    
        vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
        vert_2D = vert_2D / vert_2D[2, :]
        clr = colors[np.mod(k, len(colors))]
#        for e in edges.T:
#            ax1.plot(vert_2D[0, e], vert_2D[1, e], color=clr)
            
        x_min = np.min(vert_2D[0, :])
        x_max = np.max(vert_2D[0, :])
        y_min = np.min(vert_2D[1, :])
        y_max = np.max(vert_2D[1, :])
        if x_min <= 0:
            x_min = 1.0
            if x_max <= 0:
                x_max = 2.0
        if y_min <= 0:
            y_min = 1.0
            if y_max <= 0:
                y_max = 2.0
        if x_max >= 1914:
            x_max = 1913.0
            if x_min >= 1914:
                x_min = 1912.0
        if y_max >= 1052:
            y_max = 1051.0
            if y_min >= 1052:
                y_min = 1050.0
        
        #cv2.rectangle(rgbImage, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,0), 1)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        x_center_nor = x_center / 1914
        y_center_nor = y_center / 1052
        width_nor = width / 1914
        height_nor = height / 1052
        
        x_center_nor = format(x_center_nor, '.6f')
        y_center_nor = format(y_center_nor, '.6f')
        width_nor = format(width_nor, '.6f')
        height_nor = format(height_nor, '.6f')
        
    #cv2.imshow(f'output{ii}', rgbImage)
    
    ## save ##
    valid_path = path_tail.replace('/', '__')
    label_path = "../mydata/labels/train/" + valid_path + ".txt"
    file_handle = open(label_path, mode='w')
    # file_handle.writelines(['1',' ', '2',' ','3'])
    file_handle.writelines([training_example[1], ' ', str(x_center_nor), ' ',str(y_center_nor), ' ',str(width_nor), ' ', str(height_nor)])
    file_handle.close()
    
    fig_path = "../mydata/images/train/" + valid_path + ".jpg"
    cv2.imwrite(fig_path, rgbImage)
        

#xyz = np.fromfile(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
#xyz = xyz.reshape([3, -1])
#
#proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
#proj.resize([3, 4])






