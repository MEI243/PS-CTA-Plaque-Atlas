import numpy as np
from pathlib import Path


import pandas as pd

import os
import copy
import SimpleITK as sitk
import math
from scipy.ndimage import morphology
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def convert(o):
    # transform int64 into int
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

def distance_3d(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + 
                     (point1[1] - point2[1])**2 + 
                     (point1[2] - point2[2])**2)

def centerline_length(centerline):
    delta = np.array(centerline[0:-2]) - np.array(centerline[1:-1])
    x = np.sqrt(np.sum(np.square(delta), axis=1))
    length = np.sum(x)
    return length

def centerline_length2(centerline):
    dists = np.sqrt(np.sum(np.diff(centerline, axis=0)**2, axis=1))
    length = np.sum(dists)
    return length

def world_to_image(world_coordinates, itk_image):
    # 获取图像的方向、原点和间距
    # direction = itk_image.GetDirection()
    origin = itk_image.GetOrigin()
    spacing = itk_image.GetSpacing()

    image_index = []
    ind_index = []
    for i in range(len(world_coordinates)):
        image_index.append((np.array(world_coordinates[i])-origin)/spacing)
        ind_index.append(((np.array(world_coordinates[i])-origin)/spacing+0.5).astype(int))

    return image_index, ind_index

def get_minimum_bounding_box(centerline):
    min_x = float('inf')
    min_y = float('inf')
    min_z = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    max_z = float('-inf')
    # 寻找最小包围框的左下角和右上角坐标
    for point in centerline:
        x, y, z = point
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)

        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)
    
    return (min_x, min_y, min_z), (max_x, max_y, max_z)


def find_centerline_segment(centerline, start_point, end_point):
    min_start_distance = math.inf
    min_end_distance = math.inf
    start_index = None
    end_index = None

    # 寻找距离起始点和终止点最近的点
    for i, point in enumerate(centerline):
        start_distance = math.dist(start_point, point)
        end_distance = math.dist(end_point, point)

        if start_distance < min_start_distance:
            min_start_distance = start_distance
            start_index = i

        if end_distance < min_end_distance:
            min_end_distance = end_distance
            end_index = i

    # 根据起始点和终止点之间的点的顺序获取中心线段
    if start_index < end_index:
        segment = centerline[start_index:end_index + 1]
    else:
        segment = centerline[start_index:] + centerline[:end_index + 1]

    return segment

def ctl_dilation(ctl_pointset, mask_withpla, itk_mask, dilation_r=3): # zyx
    # 创建一个三维坐标系
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    # 创建一个示例的三维中心线图像，其中中心线的路径被标记为1
    centerline_image = np.zeros(mask_withpla.shape, dtype=np.uint8) # zyx
    for p in ctl_pointset:
        ax1.scatter(p[0], p[1], p[2], c='r', marker='o')
        p_ind = ((np.array(p) - itk_mask.GetOrigin())/ itk_mask.GetSpacing()+np.array([0.5, 0.5, 0.5])).astype(int)
        centerline_image[p_ind[2], p_ind[1], p_ind[0]] = 1

    # 定义自定义的三维管状结构元素，适合沿中心线膨胀
    struct_element = np.ones((dilation_r, dilation_r, dilation_r), dtype=np.uint8)
    # struct_element[1, 1, :] = 1

    # 使用管状结构元素沿中心线进行膨胀操作
    dilated_centerline = morphology.binary_dilation(centerline_image, structure=struct_element)

    # 获取中心线路径上的坐标
    z1, y1, x1 = np.where(dilated_centerline == 1)
    
    # 绘制中心线路径
    
    ax2.scatter(x1,y1,z1, c='b', marker='o')
    
    # 设置坐标轴标签
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 显示图像
    plt.show()
    dil_set = []
    for i in range(x1.shape[0]):
        dil_set.append([x1[i], y1[i], z1[i]])
    
    return dil_set

def toitk(np_arr_, ref_itk_):
    itk_img = sitk.GetImageFromArray(np_arr_)
    itk_img.SetDirection(ref_itk_.GetDirection())
    itk_img.SetOrigin(ref_itk_.GetOrigin())
    itk_img.SetSpacing(ref_itk_.GetSpacing())

    return itk_img


def maxConnectArea(itk_image_,first_num=2):
    """ 获取最大连通域
    return: itk image"""
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    output_connected = cc_filter.Execute(itk_image_)
    # -> 0,1,2,....一系列的连通区域编号, 0表示背景
    output_connected_array = sitk.GetArrayFromImage(output_connected)
    # print(np.unique(output_connected_array))
    num_connected_label = cc_filter.GetObjectCount()

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_connected)

    max_area = 0
    max_label_idx = 0
    # -> 找出最大的area
    # 连通域label从1开始, 0表示背景
    areas = []
    for i in range(1,num_connected_label+1):
        cur_area = lss_filter.GetNumberOfPixels(i)
        areas.append(cur_area)
        # if cur_area > max_area: 
        #     max_area = cur_area
        #     max_label_idx = i
    id = sorted(range(len(areas)), key=lambda k:areas[k], reverse=True)
    select_idx = id[0:first_num]
    
    re_mask = np.zeros_like(output_connected_array,dtype='uint8')
    for lidx in select_idx:
        re_mask[output_connected_array==lidx+1] = 1
    
    re_image = sitk.GetImageFromArray(re_mask)
    re_image.SetDirection(itk_image_.GetDirection())
    re_image.SetSpacing(itk_image_.GetSpacing())
    re_image.SetOrigin(itk_image_.GetOrigin())
    return re_image

def allConnectArea(itk_pla_mask_, itk_orig_mask):
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    output_connected = cc_filter.Execute(itk_pla_mask_)
    # -> 0,1,2,....一系列的连通区域编号, 0表示背景
    output_connected_array = sitk.GetArrayFromImage(output_connected)
    # print(np.unique(output_connected_array))
    num_connected_label = cc_filter.GetObjectCount()

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_connected)
    # max_area = 0
    # max_label_idx = 0
    # -> 找出最大的area
    # 连通域label从1开始, 0表示背景
    areas = []
    bounding_boxes = []
    for i in range(1,num_connected_label+1):
        cur_area = lss_filter.GetNumberOfPixels(i)
        bounding_box = lss_filter.GetBoundingBox(i) # [xstart, ystart, zstart, xsize, ysize, zsize]
        areas.append(cur_area)
        bounding_boxes.append(bounding_box)

    id = sorted(range(len(areas)), key=lambda k:areas[k], reverse=True)
    
    orig_mask = sitk.GetArrayFromImage(itk_orig_mask)
    for lidx in id:
        re_mask = np.zeros_like(output_connected_array,dtype='uint8')
        bdind = bounding_boxes[lidx]
        re_mask[bdind[2]:bdind[2]+bdind[5], bdind[1]:bdind[1]+bdind[4], bdind[0]:bdind[0]+bdind[3]] = np.ones((bdind[3], bdind[4], bdind[5]))

        re_mask = np.zeros_like(output_connected_array,dtype='uint8')
    
    re_image = sitk.GetImageFromArray(re_mask)
    re_image.SetDirection(itk_image_.GetDirection())
    re_image.SetSpacing(itk_image_.GetSpacing())
    re_image.SetOrigin(itk_image_.GetOrigin())
    return re_image
