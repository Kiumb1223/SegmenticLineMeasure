#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :frame2Img.py
:Description:
    1.实现将视频中的每一帧提取出来并保存到.before文件夹中,初步实现数据集的制作
                                (注意会提前将before文件家中的所有文件删掉)
:EditTime   :2023/08/07 15:37:59
:Author     :Kiumb
'''


import os
import glob 
import cv2 as cv 
import numpy as np 
from tqdm import tqdm

videoPath      = r'DatasetProcess\video'
saveFramePath  = r'DatasetProcess\before'
interval_frame = 10 

if __name__ == '__main__':
    frameId  = 0
    frameCnt = 0
    videoList = glob.glob(os.path.join(videoPath,'*.mp4'))
    if not os.path.exists(saveFramePath):
        os.mkdir(saveFramePath)
    elif os.listdir(saveFramePath) != []:
        # remove all the files in the saveFramePath folder 
        for filename in os.listdir(saveFramePath):
            os.remove(os.path.join(saveFramePath,filename))
    if videoList == []:
        raise ValueError('No videos in the specified folder')
    
    for video in tqdm(videoList,desc='ProfessBar'):
        videoCapture = cv.VideoCapture(video)
        while(videoCapture.isOpened()):
            ret , frame = videoCapture.read()
            if ret == True:
                if frameCnt % interval_frame == 0:
                    # 每interval_frame帧保存图片
                    cv.imwrite(os.path.join(saveFramePath,f'{frameId}.jpg'),frame)
                    frameId += 1
                frameCnt += 1
            else:
                videoCapture.release()
