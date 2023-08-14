#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :UnetStructure.py
:Description:
    1.构建UNET网络框架
        Encoder部分为    VGG16网络的卷积部分
        Decoder部分为    特征融合以及长采样操作
        FinalLayer部分为 整合输出通道
    2.默认使用VGG的预训练模型,所以可以使用冻结训练策略来训练UNER
:EditTime   :2023/08/03 19:55:19
:Author     :Kiumb
'''

import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self,in_channels,num_classes = 1000,init_weights = False):
        super(VGG,self).__init__()
        self.features  = nn.Sequential(
            # 64, 64, 'M'
            nn.Conv2d(in_channels,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            # 128, 128, 'M'
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            # 256, 256, 256, 'M'
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            # 512, 512, 512, 'M'
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            # 512, 512, 512, 'M'
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7,4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096,num_classes)
        )
        
        if init_weights:
            # 初始化参数
            self._initialize_weights()
    def forward(self,input):
        # 返回用于特征融合的feature map
        feature1 = self.features[:4](input)
        feature2 = self.features[4:9](feature1)
        feature3 = self.features[9:16](feature2)
        feature4 = self.features[16:23](feature3)
        feature5 = self.features[23:-1](feature4)
        return [feature1,feature2,feature3,feature4,feature5]
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

# 构造UNET的encoder部分
def VGG16(pretrained,in_channels,**kwargs):
    model = VGG(in_channels=in_channels,**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    del model.avgpool
    del model.classifier

    return model

# 构造UNET中Decoder部分
class UNETUP(nn.Module):
    def __init__(self,in_size,out_size) -> None:
        super(UNETUP,self).__init__()
        self.conv1 = nn.Conv2d(in_size,out_size,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(out_size,out_size,kernel_size=3,padding=1)
        self.up    = nn.UpsamplingBilinear2d(scale_factor = 2 )
        self.relu  = nn.ReLU(inplace = True)
    def forward(self,inputs1,inputs2):
        outputs = torch.cat([inputs1,self.up(inputs2)],dim=1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs
    
class UNETSKELETON(nn.Module):
    def __init__(self,pretrained,num_classes) -> None:
        super(UNETSKELETON,self).__init__()
        # 训练的是RGB图像，所以默认输入通道为3
        self.vgg = VGG16(pretrained=pretrained,in_channels=3)

        #上采样时输入输出通道数
        in_filter  = [192, 384, 768, 1024]
        out_filter = [64,128,256,512]

        self.up_concat4 = UNETUP(in_filter[3],out_filter[3])
        self.up_concat3 = UNETUP(in_filter[2],out_filter[2])
        self.up_concat2 = UNETUP(in_filter[1],out_filter[1])
        self.up_concat1 = UNETUP(in_filter[0],out_filter[0])

        self.finalLayer = nn.Conv2d(out_filter[0],num_classes,kernel_size = 1)
    def forward(self,input):
        [feat1,feat2,feat3,feat4,feat5] = self.vgg.forward(input)

        up4 = self.up_concat4(feat4,feat5)
        up3 = self.up_concat3(feat3,up4)
        up2 = self.up_concat2(feat2,up3)
        up1 = self.up_concat1(feat1,up2)

        res = self.finalLayer(up1)

        return res
    
    # 冻结阶段
    def freeze_vgg(self):
        for param in self.vgg.parameters():
            param.requires_grad = False
    # 解冻阶段
    def unfreeze_vgg(self):
        for param in self.vgg.parameters():
            param.requires_grad = True
