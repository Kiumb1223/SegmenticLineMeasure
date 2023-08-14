#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :sort_rename.py
:Description:
    1.重新排序并且重新命名
    (适用于制作标签过程中,将部分图片舍弃掉,此时需要再对所有的图片及其json文件进行重命名)
:EditTime   :2023/08/11 08:57:06
:Author     :Kiumb
'''
import os

prefix = 384
renamePath = r'DatasetProcess\before'
if __name__ == '__main__':
    jpgFile = [int(file.split('.jpg')[0]) for file in os.listdir(renamePath) if file.endswith('.jpg')]
    for filename in jpgFile:
        os.rename(os.path.join(renamePath,f'{filename}.jpg'),os.path.join(renamePath,f'{prefix+filename}.jpg'))
        os.rename(os.path.join(renamePath,f'{filename}.json'),os.path.join(renamePath,f'{prefix+filename}.json'))

