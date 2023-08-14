#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :voc_annotation.py
:Description:进行数据集的划分
            - trainval_percent: 选取的训练集和验证集的总数在数据集中的占比;所以测试集的占比为1 - trainval_percent 
            - train_percent:在划分好训练集和验证集的总数之后,训练集和验证集的占比则是 train_percent : 1- train_percent
:EditTime   :2023/08/12 20:12:14
:Author     :Kiumb
'''

import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
#   想要增加测试集修改trainval_percent 
#   修改train_percent用于改变验证集的比例 9:1
#-------------------------------------------------------#
trainval_percent    = 0.9
train_percent       = 0.9
if __name__ == "__main__":
    random.seed(10)
    print("Generate txt in ImageSets.")
    # 标签文件夹
    segfilepath       = r'.\datasets\SegmentationClass'
    # 保存路径
    saveBasePath      = r'.\datasets\Segmentation'
    
    # 枚举标签文件夹中的所有文件
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        # 筛选后缀为png的图片
        if seg.endswith(".png"):
            total_seg.append(seg)

    num     = len(total_seg)  
    list    = range(num)                # 筛选后的所有文件个数
    tv      = int(num*trainval_percent) # 用于当作训练集和验证集的数量
    tr      = int(tv*train_percent)     # 用于训练集的个数
    # random.sample(population,k) usage 
    # choose k unique random elements from a population sequence or list 
    trainval= random.sample(list,tv)    # 选择tv个用于训练集和验证集合
    train   = random.sample(trainval,tr)# 再从其中选择tr个用于训练集  
    
    print("train and val size",tv)
    print("traub suze",tr)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        # for example
        # ‘2007_002896.png’
        # [:-4] - > 2007_002896
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")
    classes_nums        = np.zeros([256], np.int8)
    for i in tqdm(list):
        # 和for i in list 相比，
        # i的枚举值都是相同的，不同的是会加上进度条
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。"%(png_file_name))
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。"%(name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。"%(name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print("JPEGImages中的图片应当为.jpg文件、SegmentationClass中的图片应当为.png文件。")
    print("如果格式有误，参考:")
    print("https://github.com/bubbliiiing/segmentation-format-fix")