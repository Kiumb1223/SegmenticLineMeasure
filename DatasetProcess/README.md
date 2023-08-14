# Readme
> DatasetProcess的作用是数据集的制作

现按照数据集的制作过程来逐一介绍每个文件(文件夹)在其中的作用。
1. 我们先将采集到的视频文件保存到**video**文件夹中,然后运行**frame2Img.py**文件(可以根据frame2img.py中的header信息来适当调整参数)，最后产生的每帧图片都保存到**before**文件夹中；

2. 得到每帧图片之后，我们在运行labelme来进行标签的制作。制作过程中，如果对部分原图片不符合而删去，可以运行**sort_rename.py**程序来对before文件夹进行整理。最后完成标签的制作时，before文件夹中会有每张图片及其对应的json文件；

   ```
   pip install labelme==3.16.7
   # 我安装的版本是3.16.7
   ```

3. 之后运行**json_to_dataset.py**文件来输出标签图片，同时还将图片和标签图片额外保存到主目录的dataset文件下的JPEGImages和SegmentationClass文件夹下(训练文件所指向的数据集路径就是这两个文件夹，所以不需要再进行额外的复制粘贴)；

4. 之后运行**voc_annotation.py**实现数据集的划分，划分的比例在该文件的header文件有所说明，按需调整参数即可。最后输出的文本文件则会保存到父目录的dataset文件下的Segmentation文件夹下(该文件夹同时所被训练文件所公用)；

5. 最后可以运行**genPredict.py**文件，从上一步划分出的测试集中提取图片到**predict**文件夹中，为之后的预测做好准备。