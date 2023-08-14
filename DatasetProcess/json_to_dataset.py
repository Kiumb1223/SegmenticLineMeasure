#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :json_to_dataset.py
:Description:
    使用labelme制作完标签之后,将json文件转变成标签文件,
    并且将原图和标签图片分别保存到dataset文件夹下的JPEGImages和SegmentationClass文件夹下
:EditTime   :2023/08/12 20:16:39
:Author     :Kiumb
'''

import os
import json
import base64
import PIL.Image
import numpy as np
import os.path as osp
from labelme import utils

'''
阅读完该段代码的笔记：
1. 语义分割中，每个像素的标签值通常是一个整数，这个整数则是classes列表中的对应标签的索引值
2. 变量new代表的就是语义分割任务下的单通道标签图
3. 为了可视化，变量new经过颜色映射后变成了三通道的图片
'''

if __name__ == '__main__':
    jpgs_path   = "./datasets/JPEGImages"
    pngs_path   = "./datasets/SegmentationClass"
    classes     = ["_background_","pipe"]
    
    count = os.listdir(r"DatasetProcess\before") 
    for i in range(0, len(count)):
        
        path = os.path.join(r"DatasetProcess\before", count[i])

        if os.path.isfile(path) and path.endswith('json'):
            # 遍历每个json文件，并进行操作
            data = json.load(open(path))
            # 读取图片数据，保存到img上
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)

            # 构造标签的键值对
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:    
                label_name = shape['label']
                if label_name in label_name_to_value:
                    # 当标签已出现过
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            
            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))
            
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
            
                
            PIL.Image.fromarray(img).save(osp.join(jpgs_path, count[i].split(".")[0]+'.jpg'))

            new = np.zeros([np.shape(img)[0],np.shape(img)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)   # 这里才是根据classes列表中的各个物体的键值对反映到标签当中去
                new = new + index_all*(np.array(lbl) == index_json)

            utils.lblsave(osp.join(pngs_path, count[i].split(".")[0]+'.png'), new)
            print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')
