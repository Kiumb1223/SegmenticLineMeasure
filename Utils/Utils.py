#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :Utils.py
:Description:
    集成化一些函数
:EditTime   :2023/08/05 11:31:19
:Author     :Kiumb
'''

import numpy as np
from PIL import Image

def cvtColor(img):
    '''
    :Description:将PIL格式的图像转换为RGB图像(网络的输入只支持RGB图像)
    :Parameter  img:PIL格式图片
    :Return     img:转换成RGB的图片数据
    '''
    if (len(np.shape(img)) == 3) and np.shape(img)[2] == 3 : 
        return img
    else:
        img = img.convert("RGB")
        return img
    
def get_lr(optimizer):
    '''
    :Description:获取学习率
    :Parameter  optimizer:优化器
    :Return     param_group['lr']:学习率
    '''
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh