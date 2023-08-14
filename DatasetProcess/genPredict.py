#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :genPredict.py
:Description:
    1. 将用于测试的图片另存到predict文件夹中去
:EditTime   :2023/08/10 15:43:16
:Author     :Kiumb
'''
import os 
import shutil
from tqdm  import tqdm 
testTXT = r'datasets\Segmentation\test.txt'
predictDir = r'DatasetProcess\predict'
jpgPath = r'datasets\JPEGImages'
if __name__ == '__main__':
    with open(testTXT,'r') as f:
        tmpList = f.readlines()
    # 去除\n
    predict_idx = [name.split()[0] for name in tmpList]

    if not os.path.isdir(predictDir):
        os.mkdir(predictDir)
    
    
    for file in tqdm(os.listdir(jpgPath)):
        if file.split('.jpg')[0] in predict_idx:
            shutil.copy2(os.path.join(jpgPath,file),os.path.join(predictDir,file))