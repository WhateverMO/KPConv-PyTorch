
import numpy as np
from utils.ply import *
import colorsys, random, os, sys
import pickle
import os
from os.path import exists, join, isdir

from utils.ply import read_ply,write_ply


def random_colors(N, bright=True, seed=0):
    brightness = 1.0 if bright else 0.7
    hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(seed)
    random.shuffle(colors)
    return colors


def label_to_color(origin_xyz=None,train_points_path=None,rgb_codes=None):
    print('label to color start')
    if origin_xyz is None:
        origin_xyz=r'D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\test_H3D\kpconv\val_preds_160'
    if train_points_path is None:
        train_points_path=r'D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\test_H3D\kpconv\val_preds_160'
    out_path=os.path.join(train_points_path,'color')
    error_path=os.path.join(train_points_path,'error')
    true_path=os.path.join(train_points_path,'true')
    os.mkdir(error_path) if not exists(error_path) else None
    os.mkdir(out_path) if not exists(out_path) else None
    os.mkdir(true_path) if not exists(true_path) else None
    points=[f for f in os.listdir(train_points_path) if f[-4:]=='.ply']
    for each_file in points:
        origin_XYZ=read_ply(os.path.join(origin_xyz,each_file))
        original_ply=read_ply(os.path.join(train_points_path,each_file))
        cloud_x=origin_XYZ['x']
        cloud_y=origin_XYZ['y']
        cloud_z=origin_XYZ['z']
        #ADD FEATURE

        label=original_ply['preds']
        label_true=origin_XYZ['class']
        label_error=label-label_true
        label_error=label_error.reshape(len(label_error),1)
        cloud_x=cloud_x.reshape(len(cloud_x),1)
        cloud_y=cloud_y.reshape(len(cloud_y),1)
        cloud_z=cloud_z.reshape(len(cloud_z),1)
        label=label.reshape(len(label),1)
        label_true = label_true.reshape(len(label_true), 1)
        cloud_x=cloud_x.astype(np.float32)
        cloud_y=cloud_y.astype(np.float32)
        cloud_z=cloud_z.astype(np.float32)

        label=label.astype(np.int32)
        label_true = label_true.astype(np.int32)
        label_error=label_error.astype(np.int32)
        cloud_points=np.hstack((cloud_x,cloud_y,cloud_z,label))
        cloud_points_eoor=np.hstack((cloud_x,cloud_y,cloud_z,label_error))
        cloud_points_true=np.hstack((cloud_x,cloud_y,cloud_z,label_true))

        data1 = cloud_points[:,:3]
        label1 = cloud_points[:,3]

        data2=cloud_points_eoor[:,:3]
        label2=cloud_points_eoor[:,3]

        data3 = cloud_points_true[:, :3]
        label3 = cloud_points_true[:, 3]

        if rgb_codes is None:
            raise ValueError('Please provide the RGB color codes for each class.')
        elif type(rgb_codes) is not list:
            if type(rgb_codes) is int:
                rgb_codes = random_colors(rgb_codes)
            else:
                raise ValueError('Please provide the RGB color codes for each class as a list.')
        
        # rgb_codes = [[200, 90, 0],
        #             [255, 0, 0],
        #             [255, 0, 255],
        #             [0, 220, 0],
        #             [0, 200, 255]]
        # rgb_codes = [[190,190,190],   #DFC2019
        #              [46,139,87],
        #              [255,97,0],
        #              [0,0,255],
        #              [255,255,0]]
        rgb_codes = [[192,192,192],  # LASDU #ground
                     [0,0,255],#building
                     [0,100,0],#tree
                     [152,251,152],#low veg
                     [255,69,0]]#artifact
        # rgb_codes =[[0, 0, 255], #ISPRS
        #             [152, 245, 255],
        #             [190, 190, 190],
        #             [255, 99, 71],
        #             [255, 0, 255],
        #             [255, 0, 0],
        #             [255, 255, 0],
        #             [0,255,0],
        #             [46,139,87]]
        # rgb_codes = [[178, 203, 47],  # H3D
        #             [183, 179, 170],
        #             [33, 151, 163],
        #             [168, 34, 107],
        #             [255, 122, 89],
        #             [254, 215, 136],
        #             [89, 125, 53],
        #             [0, 128, 65],
        #             [170, 86, 0],
        #             [253, 255, 6],
        #             [128, 0,0]]
        # rgb_codes = random_colors(6)
        color = np.zeros((label1.shape[0], 3))
        color_error=np.zeros((label2.shape[0], 3))
        color_true=np.zeros((label3.shape[0],3))
        for i in range(label1.shape[0]):

            color[i,:] = [code for code in rgb_codes[int(label1[i])]]
            # color = color.astype(np.uint8)
        color = color.astype(np.uint8)
        print(color.shape)
        write_ply(os.path.join(out_path,each_file),[data1,color,label1],
                ['x','y','z','red','green','blue','class'])

        for i in range(label1.shape[0]):

            color_true[i,:] = [code for code in rgb_codes[int(label3[i])]]
            # color = color.astype(np.uint8)
        color_true = color_true.astype(np.uint8)
        print(color.shape)
        write_ply(os.path.join(true_path,each_file),[data3,color_true,label3],
                ['x','y','z','red','green','blue','class'])


        # Error Maping
        for j in range(label2.shape[0]):
            if int(label2[j])==0:
                color_error[j,:] = [135, 206, 250] #LightSkyBlue
            else:
                # color_error[j, :] = [138, 54, 15]
                color_error[j, :] = [255, 0, 0]
        color_error = color_error.astype(np.uint8)
        print(color.shape)
        write_ply(os.path.join(error_path,each_file), [data2, color_error, label2],
                ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
    print('label to color end')
    print()
        
if __name__ == '__main__':
    label_to_color()