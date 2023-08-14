#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :Dataloader.py
:Description:定义UNet的Dataset类,喂入网络的数据都经过数据增强的处理
:EditTime   :2023/08/05 10:57:32
:Author     :Kiumb
'''
import os 
import cv2 as cv
import numpy as np
from PIL import Image
import torch
from Utils.Utils import cvtColor
from torch.utils.data.dataset import Dataset

class UNETDATASET(Dataset):
    '''
    :Description:UNet的数据加载类
    :Args annotation:指定的数据名称
          input_shape:指定的输出尺寸
          num_classes:分类的数量
          bl_train:bool型变量,表征载入的数据是否为训练集数据
          dataset_path:载入的数据集名称
    '''
    def __init__(self,annotations,input_shape,num_classes,bt_train,dataset_path) -> None:
        super().__init__()
        self.annotations = annotations 
        self.length      = len(annotations)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.bt_train    = bt_train
        self.dataset_path= dataset_path
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 读入每张数据以及标签
        name = self.annotations[index].split()[0]
        jpg = Image.open(os.path.join(self.dataset_path,"JPEGImages",name + ".jpg"))
        png = Image.open(os.path.join(self.dataset_path,"SegmentationClass",name + ".png"))

        jpg,png = self._augmentData(jpg,png,self.bt_train)
        # 对图片进行归一化,并进行维度变换dim ->[channel,height,width]
        jpg = np.transpose(np.array(jpg,np.float64) / 255.0,[2,0,1])
        # png格式为PIL,因为需进行独热编码,所以需要转换成ndarray格式
        png = np.array(png)   # [512,512]
        #---------------------------------#
        # 转换成one_hot形式
        #---------------------------------#
        seg_labels = np.eye(self.num_classes + 1 )[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]),int(self.input_shape[1]),self.num_classes + 1 ))
        return jpg,png,seg_labels
    
    def _rand(self,a = 0,b = 1):
        return np.random.rand() *(b-a) + a 
    
    def _augmentData(self,image,label,bt_train,
                     jitter=.3,hueJitter = .1,
                     satJitter=.7,valJitter=.3):
        '''
        :Description:内部函数,只用于数据增强
        :Parameter  image:输入到UNet的图片
                    label:对应的标签
                    bt_train:喂入的图片是否用于测试
                    jitter:用于对图片的缩放以及宽高比产生特定的扰动
                    hueJitter&satJitter&valJitter:对图片的h、s、v三通道分别添加扰动
        :Return     image:经过图像增强过的图片,其尺寸满足input_shape(np.ndarray类型)
                    label:尺寸满足input_shape的标签(PIL.Image.Image   palette类型的图片)
        '''
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        iw,ih = image.size
        h,w   = self.input_shape
        if not bt_train:
            # 当前数据集不用于训练
            # 所以只进行添加灰度条以保持宽高比，并且缩放到input_shape尺寸即可
            scale = min(h/ih,w/iw)
            nh    = int(scale * ih)  
            nw    = int(scale * iw)  
            image = image.resize((nw,nh),Image.BICUBIC)
            imgNew= Image.new('RGB',[w,h],(128,128,128))
            imgNew.paste(image,((w-nw)//2,(h-nh)//2))

            label = label.resize((nw,nh),Image.NEAREST)
            labelNew = Image.new('L',[w,h],(0))
            labelNew.paste(label,((w-nw)//2,(h-nh)//2))

            return imgNew,labelNew
        # 当前数据用于训练


        #---------------------------------#
        # 对图像进行随机缩放(缩放0.25-2倍)，并且对宽高比添加扰动(缩放0.7-1.3倍)
        #---------------------------------#
        aspectRatioRand = iw / ih * self._rand(1-jitter,1+jitter) / self._rand(1-jitter,1+jitter)
        scale = self._rand(0.25,2)

        if aspectRatioRand >= 1:
            # 新图像的宽度大于高度
            nw = int(scale*iw)
            nh = int(nw/aspectRatioRand)
        else:
            # 新图像的宽度小于高度
            nh = int(scale*ih)
            nw = int(aspectRatioRand*nh)
        
        image = image.resize((nw,nh),Image.BICUBIC)
        label = label.resize((nw,nh),Image.NEAREST)
        
        #---------------------------------#
        # 随机选择是否要翻转图像
        #---------------------------------#
        bt_flip = self._rand()
        if bt_flip > .5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        #---------------------------------#
        # 添加灰度条，以保证尺寸满足input_shape 
        #---------------------------------#
        imgNew   = Image.new('RGB',(w,h),(128,128,128))
        labelNew = Image.new('L',(w,h),(0))

        # 此处应用到Image.paste函数，会自动舍弃超出部分
        # 所以不用很在意offsetX和offsetY的符号问题
        offsetX  = int(self._rand(0,w-nw))
        offsetY  = int(self._rand(0,h-nh))
        imgNew.paste(image,(offsetX,offsetY))
        labelNew.paste(label,(offsetX,offsetY))
        image = imgNew
        label = labelNew

        #---------------------------------#
        # ColorJitter
        # 在HSV色彩空间，对图像的三个通道添加扰动
        #---------------------------------#
        img_data    = np.array(image,np.uint8)
        rand        = np.random.uniform(-1,1,3) * [hueJitter,satJitter,valJitter] + 1
        hue,sat,val = cv.split(cv.cvtColor(img_data,cv.COLOR_RGB2HSV))
        dtype       = img_data.dtype
        # 构造查找表
        x           = np.arange(0,256,dtype=rand.dtype)
        lut_hue     = ((x*rand[0]) % 180).astype(dtype)
        lut_sat     = np.clip(x*rand[1],0,255).astype(dtype)
        lut_val     = np.clip(x*rand[2],0,255).astype(dtype)
        image_data  = cv.merge((cv.LUT(hue,lut_hue),cv.LUT(sat,lut_sat),cv.LUT(val,lut_val)))
        image_data  = cv.cvtColor(image_data,cv.COLOR_HSV2RGB)

        return image_data,label


# if __name__ == '__main__':
#     input_shape = [512,512]
#     dataset_path = r'E:\Desktop\MyProj\MeasuringSize\solution3\datasets'
#     num_class = 2
#     trainSet_path = os.path.join(dataset_path,'Annotations','train.txt')
#     with open(trainSet_path,mode='r') as f:
#         annotaions = f.readlines()
#     UnetDataset = UNETDATASET(annotaions,input_shape,num_class,True,dataset_path)
#     image,label = UnetDataset[0]
#     image = Image.fromarray(image)
#     image.show()    
#     label.show()    

# DataLoader中collate_fn使用
# 用来处理不同情况下的dataset封装
def unet_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    # 将列表转换成tensor类型
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels