#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File       :cornerDetect.py
@Description:角点检测算法
    读取DatasetProcess文件夹下的predict的图片,喂入UNet网络,并将生成的分割图进行角点检测,最后在进行标注 
@EditTime   :2023/07/31 09:28:56
@Author     :Kiumb
'''

import os 
import cv2 as cv 
import numpy as np 
from PIL import Image
from tqdm import tqdm
from Net.Unet import UNET 

def compute_rectangle_corner(myCorner):
    '''
    :Description:将矩形的角点坐标进行排序,并计算出中心线的坐标及长度
        排序结果为：
                            centerLine[0]
            myCornerNew[0] --------------- myCornerNew[1]
                |                               |
                |                               |
            myCornerNew[2] --------------- myCornerNew[3]
                            centerLine[1]
        
        Attention:排好序的坐标并不是按照长宽来排的。其次,所计算的disatance一直都是竖直方向的距离

    :Parameter  :myCorner    - 待排序的坐标
    :Return     :myCornerNew - 排好序的坐标
                 centerLine  - 中心线坐标
                 distance    - 中心线长度(欧式距离)
    '''
    myCornerNew = np.zeros_like(myCorner)
    centerLine = np.zeros((2,2))

    resRowSum = np.sum(myCorner,axis=1)
    myCornerNew[0] = myCorner[np.argmin(resRowSum)]
    myCornerNew[3] = myCorner[np.argmax(resRowSum)]
    resRowDiff = np.diff(myCorner,axis=1)
    myCornerNew[1] = myCorner[np.argmin(resRowDiff)]
    myCornerNew[2] = myCorner[np.argmax(resRowDiff)]

    centerLine[0,:] = np.mean(myCornerNew[:2,:],axis=0)
    centerLine[1,:] = np.mean(myCornerNew[2:4,:],axis=0)
    centerLine = centerLine.astype(np.int32)
    
    distance = np.sqrt(np.sum(np.diff(centerLine,axis=0) ** 2))
    return myCornerNew,centerLine[:4,:],distance

def find_approx_rectangle_corners(img2,minArea = 500,presicion = 0.01,filter = 0,bt_show = False):
    '''
    :Description:寻找角点坐标
    :Parameter  img2:待寻找角点坐标的图片(OpenCV读取)
                minArea:按照面积的大小进行一个初步的筛选
                presicion:多边形拟合的一个参数
                filter:用于筛选的指标,代表着筛选的形状的角点数,设置为0则不用于筛选
                bt_show:布尔值,设True则会显示角点
    :Return     finalContours[0][2]:返回的最大面积的顶点坐标
    '''
    img2Gray    = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    contours,_ = cv.findContours(img2Gray,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    finalContours = []
    for idx,contour in enumerate(contours):
        area = cv.contourArea(contour)
        if area > minArea:
            epsilon = presicion * cv.arcLength(contour,True)
            approx = cv.approxPolyDP(contour,epsilon,True)
            bbox = cv.boundingRect(approx)
            approx = np.squeeze(approx)
            if filter > 0:
                if len(approx) == filter:
                    # print(f'idx:{idx};length:{len(approx)}')
                    finalContours.append([len(approx),area,approx,contour])
            else:
                finalContours.append([len(approx),area,approx,bbox,contour])
    # 按照面积大小排序，默认面积最大的是目标物体
    finalContours = sorted(finalContours,key = lambda x:x[1] ,reverse= True)
    if bt_show:
        for con in finalContours:
            cv.drawContours(img2,con[4],-1,(255,255,255),1)
            cv.imshow('DrawContours',img2)
            cv.waitKey(0)
            cv.destroyAllWindows()
    return finalContours[0][2]


if __name__ == '__main__':


    filePath = r'DatasetProcess\predict'
    savePath = r'Output\picAnnotation'

    unet = UNET()
    bt_rotate = False 
    for filename in tqdm(os.listdir(filePath)):
        img  = Image.open(os.path.join(filePath,filename))
        img2 = unet.detect_image(img,bt_blend=False)

        img  = np.array(img)
        img  = cv.cvtColor(img,cv.COLOR_RGB2BGR)
        img2 = np.array(img2)
        img2  = cv.cvtColor(img2,cv.COLOR_RGB2BGR)
        
        if img2.shape[0] < img2.shape[1]:
            # 如果是高度小于宽度，则进行逆时针翻转
            # 采集到的图片特性所决定这样子处理
            img2 = cv.rotate(img2,cv.ROTATE_90_COUNTERCLOCKWISE)
            bt_rotate = True
        img2Copy = img2.copy()
        cor = find_approx_rectangle_corners(img2,bt_show=False)
        corNew,centerline,distance = compute_rectangle_corner(cor)

        # 标注
        for coor in centerline:
            cv.circle(img2Copy,coor,2,(255,255,255),thickness=2)
        for coor in corNew:
            cv.circle(img2Copy,coor,2,color=(255,255,255),thickness=2)
        cv.line(img2Copy,centerline[0],centerline[1],(255,255,255),thickness = 2)
        cv.putText(img2Copy,f'Length:{np.floor(distance)} pixel',(centerline[0] + centerline[1] )//2,cv.FONT_HERSHEY_SIMPLEX,1, (100, 200, 200), 2)
        if bt_rotate:
            # 旋转回来
            img2Copy = cv.rotate(img2Copy,cv.ROTATE_90_CLOCKWISE)
            bt_rotate = False
        
        # 混合标签和源图像
        img2Res = cv.add(img*0.6,img2Copy*0.4).astype(np.uint8)
        cv.imwrite(os.path.join(savePath,filename),img2Res)