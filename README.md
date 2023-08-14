

# Semantic Line Measure

该工程项目利用了**深度学习和传统图像处理方法**来实现了目标物体的**角点检测**以及**长度计算**。

![img_store/hstack.png at master · Kiumb1223/img_store (github.com)](https://github.com/Kiumb1223/img_store/blob/master/hstack.png)

## 开发环境

* Python  == 3.19
* Pytorch == 1.12.1 （源码中使用了fp16混精度加速，所以Pytorch版本需高于1.7.1）
* labelme == 3.16.7
*  and so on 

## 各文件介绍

* `DatasetProcess`文件夹：该文件夹集成了手工制作数据集的所有代码，可参考里面的readme来着手制作项目的数据集。在运行里面的代码时，会自动将输出的图片保存到`datasets`文件夹中，无需手动搬移。
* `datasets`文件夹：该文件夹中符合VOC数据集的格式，包含了图片、对应标签以及记录数据集划分的文本文件。在训练阶段中，读取的文件都是从该文件夹中读取。
* `logs`文件夹：该文件夹会保存训练阶段时的最优权重、最后epoch时的权重以及loss值等信息。
* `model_data`文件夹：该文件夹包含了模型训练或预测阶段所加载的权值。
* `Net`文件夹：该文件夹包含了Unet的框架实现代码、训练过程中涉及的函数和类、指标函数等。必要时，可以查看各个python文件中的header信息来进一步了解各个package的作用。
* `Utils`文件夹：该文件夹包含了一些封装好的函数工具。可查看其header信息。
* `Output`文件夹：`vedioTest`子文件夹包含的是Unet网络对于几个包含目标物体视频的预测结果；`picAnnotaion`子文件夹包含的是在测试集上应用最后算法的结果展示图。
* `cornerDetect.py`文件：该文件使用了Unet网络以及角点检测算法，从而实现了项目目标。
* `predict.py`文件：UNet网络的预测文件，具有**单张图片预测**、**指定文件夹预测**、**视频预测**这三大功能。
* `train.py`文件：UNet网络训练文件，支持冻结训练策略、多种优化器选择、多种学习率衰减策略等。

## 总结

角点检测、长度计算的精度主要在于UNet网络的分割效果。观察`vedioTest`文件夹中的预测视频发现，训练出来的网络对于垂直方向的物体分割效果比水平方向的物体分割更好。



