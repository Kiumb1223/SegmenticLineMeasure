#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :UnetMetric.py
:Description:
    1. 使用f1_—score指标来对模型进行评估
:EditTime   :2023/08/07 17:42:32
:Author     :Kiumb
'''
import torch 
import torch.nn.functional as F

def f_score(inputs,targets,beta=1,smooth=1e-5,threhold=.5):
    '''
    :Description:
        f_score的概念介绍可参考https://blog.csdn.net/saltriver/article/details/74012163
        此处β默认设置为1,代表precision和recall同样重要
    :Parameter  inputs:网络输入
                targets:独热码形式的标签
                beta&smooth&threhold:计算Fscore所需参数
    :Return     score:f_score的值
    '''
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = targets.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = targets.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score    