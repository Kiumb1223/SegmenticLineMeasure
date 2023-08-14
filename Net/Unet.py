#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :Unet.py
:Description:
    1.构建可用于预测的UNET网络类
        * detect_image 方法:只支持单张图片的检测
            根据bt_blend的值来返回混合图像或者语义分割结果图
:EditTime   :2023/08/08 16:32:54
:Author     :Kiumb
'''

import copy
import torch
import colorsys
import cv2 as cv 
import numpy as np 
from torch import nn
from PIL import Image
import torch.nn.functional as F
from Net.UnetStructure import UNETSKELETON
from Utils.Utils import show_config,cvtColor,resize_image

class UNET():
    _defaults ={
        #-------------------------------------------------------------------#
        #   model_path指向训练好的权值参数
        #-------------------------------------------------------------------#
        "model_path"    : r'model_data\best_epoch_weights.pth',
        #--------------------------------#
        #   所需要区分的类的个数+1
        #--------------------------------#
        "num_classes"   : 2,
        #--------------------------------#
        #   输入图片的大小
        #--------------------------------#
        "input_shape"   : [512, 512],
        #--------------------------------#
        #   是否使用Cuda
        #--------------------------------#
        "cuda"          : True,        
    } 
    def __init__(self,**kwargs) -> None:
        self.__dict__.update(self._defaults)
        for name,value in kwargs.items():
            setattr(self,name,value)
        # 画框设置不同的颜色
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # 产生模型
        self.generate()

        show_config(**self._defaults)

    def generate(self):
        self.net = UNETSKELETON(pretrained=False,num_classes=self.num_classes)
        device   = torch.device('cuda' if torch.cuda.is_available and self.cuda else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path,map_location=device))
        self.net = self.net.eval()
        print(f'{self.model_path} model and classes loaded')
        self.net = nn.DataParallel(self.net)
        self.net = self.net.to(device)
        
    def detect_image(self,image,bt_blend = True):
        device   = torch.device('cuda' if torch.cuda.is_available and self.cuda else 'cpu')
        # 代码仅支持RGB图像的预测，所有其他类型的图像都会转换成RGB
        image = cvtColor(image)
        imageBackup = copy.deepcopy(image)
        original_h  = np.array(image).shape[0]
        original_w  = np.array(image).shape[1]
        # 进行缩放
        image_data,nw,nh = resize_image(image,(self.input_shape[1],self.input_shape[0]))
        # 添加batch_size维度
        image_data = np.expand_dims(np.transpose(np.array(image_data,np.float32)/ 255.0,(2,0,1)),0)
        with torch.no_grad():
            images = torch.from_numpy(image_data).to(device)
            # 进行预测
            predict = self.net(images)[0]
            # 取出每一个像素点的种类
            predict = F.softmax(predict.permute(1,2,0),dim=-1).cpu().numpy()    
            # 裁剪掉灰条部分
            predict = predict[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                              int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            # 对预测图片进行resize
            predict = cv.resize(predict,(original_w,original_h),interpolation=cv.INTER_LINEAR)
            # 取出每一个像素点的种类
            predict = predict.argmax(axis = -1)

            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(predict, [-1])], [original_h, original_w, -1])
            # 将新图片转换成Image的形式
            image   = Image.fromarray(np.uint8(seg_img))  
                 
        if bt_blend:
            image   = Image.blend(imageBackup, image, 0.7)
        return image     