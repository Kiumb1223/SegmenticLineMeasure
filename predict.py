#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :predict.py
:Description:
:EditTime   :2023/08/09 16:35:25
:Author     :Kiumb
'''
import os 
import time 
import cv2 as cv 
import numpy as np
from tqdm import tqdm
from PIL import Image
from Net.Unet import UNET

if __name__ == '__main__':
    #---------------------------------#
    # mode用于指定测试的模式
    #   - 'img_predict'   表示单张图片预测
    #   - 'video_predict' 表示视频预测
    #   - 'dir_predict'   表示预测指定的文件夹下的所有图片
    #---------------------------------#
    mode            = 'video_predict'
    #---------------------------------#
    #   video_path         用于指定视频的路径，当video_path=0时表示检测摄像头
    #   video_save_path    表示视频保存的路径，当video_save_path=""时表示不保存
    #   video_fps          用于保存的视频的fps
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #---------------------------------#
    video_path      = r'DatasetProcess\video\6.mp4'
    video_save_path = r"Output\videoTest\6_unet.mp4"
    video_fps       = 25.0
    #---------------------------------#
    #   dir_origin_path    指定了用于检测的图片的文件夹路径
    #   dir_save_path      指定了检测完图片的保存路径
    #---------------------------------#
    dir_origin_path = "DatasetProcess\predict"
    dir_save_path   = ""


    # 实例化网络
    unet = UNET()

    if mode == 'img_predict':
        while True:
            path = input('Input image filename:')
            try:
                image = Image.open(path)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image,bt_blend=False)
                r_image.show()
                # r_image.save('./res.png')
    elif mode == 'video_predict':
        capture=cv.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv.VideoWriter_fourcc('M', 'P', '4', 'V')             
            size = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
            out = cv.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(unet.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv.putText(frame, "fps= %.2f"%(fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv.imshow("video",frame)
            c= cv.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
    elif mode == 'dir_predict':
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = unet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))