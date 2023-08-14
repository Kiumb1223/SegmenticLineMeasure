#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :train.py
:Description:训练网络
:EditTime   :2023/08/05 17:05:22
:Author     :Kiumb
'''

import os
import torch 
import datetime
import numpy as np
import torch.optim as optim
from Net.Dataloader import UNETDATASET 
from torch.utils.data import DataLoader
from Net.UnetStructure import UNETSKELETON
from Net.UnetTraining import LossHistory, init_lr_epoch,set_optimizer_lr,unet_dataset_collate,Train_OneEpoch

class CONFIG():
    '''
    :Description:
        创建配置自定义参数类,便于管理这部分参数
    :Args:各个属性都有其定义,不额外赘述
    '''
    def __init__(self) -> None:

        #---------------------------------#
        # 数据集路径
        # （文件结构树）datasets
        #                 | —— JPEGImages        :保存的是数据集
        #                 | —— Segmentation      :保存的是随机划分的测试集、训练集、验证集的文本文件（记录文件名）
        #                 | —— SegmentationClass :保存的是数据集中的标签
        #---------------------------------#
        self.dataset_path        = './datasets/'

        #---------------------------------#
        # 参数设置
        # classes            - 语义分割的种类
        # num_classes        - 语义分割的种类数
        # cls_weights        - 每个种类对应的损失权值
        # pretrained         - 是否载入预训练参数的标志位
        # model_path         - 模型权值的读取路径（当model_path指向训练好的权重时，pretrained自动失效，因为程序中会进行权值覆盖）
        # cuda               - 是否开启GPU训练的标志位
        # device             - 无需设置，会自动判断选择Cpu还是Gpu
        #---------------------------------#
        self.classes             = ['background','pipe']
        self.num_classes         = len(self.classes)
        self.cls_weights         = np.array([1,2], np.float32)
        self.input_shape         = [512,512]
        self.pretrained          = True
        self.model_path          = r'model_data\vgg16-397923af.pth'
        self.cuda                = True 
        self.device              = torch.device("cuda" if torch.cuda.is_available() and self.cuda else 'cpu')

        #---------------------------------#
        # 冻结训练和解冻训练参数设置 
        # Freeze_Train        - 开启冻结训练标志位 
        # Init_Epoch          - 模型当前开始的训练世代，其值可以大于Freeze_Epoch（此时会跳过解冻阶段，断点续训时使用） 
        # Freeze_Epoch        - 冻结训练的epoch数（Freeze_Train = False失效）
        # Freeze_batch_size   - 冻结训练时的batch数（Freeze_Train = False失效）
        # Unfreeze_Epoch      - 解冻训练的epoch数(实际上是训练的总epoch数)
        # Unfreeze_batch_size - 解冻训练的batch数
        #---------------------------------#
        self.Freeze_Train        = True

        self.Init_Epoch          = 0

        self.Freeze_Epoch        = 10
        self.Freeze_batch_size   = 2

        self.UnFreeze_Epoch      = 50
        self.Unfreeze_batch_size = 2

        #---------------------------------#
        # 优化器设置参数
        # optimizer_type       - 使用到的优化器种类，可选的有adam、sgd
        # momentum             - 优化器内部使用到的momentum参数
        # weight_decay         - 权值衰减，可防止过拟合（使用adam时会导致weight_decay错误，建议此时设置为0）
        # Init_lr              - 模型的最大学习率
        # Min_lr               - 模型的最小学习率
        # lr_decay_type        - 学习率下降方式，可选的有step，cos
        #---------------------------------#
        self.optimizer_type      = "adam"
        self.momentum            = 0.9
        self.weight_decay        = 0

        self.Init_lr             = 1e-4  if self.optimizer_type ==  'adam' else 1e-2
        self.Min_lr              = self.Init_lr * 0.01 
        self.lr_decay_type       = 'cos'

        #---------------------------------#
        # 参数设置
        # save_period          - 每隔save_period个epoch保存一次权值
        # save_dir             - 权值文件及其他一些过程文件的保存路径
        # time_str             - 无需设置，代表当前训练的时间戳
        # log_dir              - 每次训练产生的过程文件保存路径
        #---------------------------------#
        self.save_period        = 5
        self.save_dir           = 'logs'
        self.time_str           = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        self.log_dir            = os.path.join(self.save_dir, "loss_" + str(self.time_str))



if __name__ == '__main__':
    cfg = CONFIG()
    #---------------------------------#
    # 载入模型并加载权值
    #---------------------------------#
    model = UNETSKELETON(pretrained=cfg.pretrained,num_classes=cfg.num_classes)

    print(f"Load weight:{cfg.model_path}")
    model_dict = model.state_dict()
    train_dict = torch.load(cfg.model_path,map_location=cfg.device)
    load_key,no_load_key,tmp_dict = [],[],{}
    for k,v in train_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            tmp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)

    model_dict.update(tmp_dict)
    model.load_state_dict(model_dict)

    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    
    #---------------------------------#
    # 加载训练集和验证集
    #---------------------------------#
    with open(os.path.join(cfg.dataset_path,"Segmentation","train.txt"),mode='r') as f:
        trainSet_names = f.readlines()
    num_train = len(trainSet_names)
    with open(os.path.join(cfg.dataset_path,"Segmentation","val.txt"),mode='r') as f:
        valSet_names = f.readlines()
    num_val   = len(valSet_names)
    train_dataset = UNETDATASET(trainSet_names,cfg.input_shape,cfg.num_classes,True,cfg.dataset_path)
    val_dataset   = UNETDATASET(valSet_names,cfg.input_shape,cfg.num_classes,False,cfg.dataset_path)

    #---------------------------------#
    # 初始化部分参数
    #---------------------------------#
    batch_size,Init_lr_fit,lr_scheduler_func , epoch_step , epoch_step_val = init_lr_epoch(cfg,True,num_train,num_val)
    
    trainSet_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True,
                                 drop_last=True,collate_fn=unet_dataset_collate)
    valSet_loader   = DataLoader(val_dataset,batch_size=batch_size,shuffle=True,pin_memory=True,
                                 drop_last=True,collate_fn=unet_dataset_collate)
    
    #---------------------------------#
    # 优化器选择
    #---------------------------------#
    optimizer = {
        'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (cfg.momentum, 0.999), weight_decay = cfg.weight_decay),
        'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = cfg.momentum, nesterov=True, weight_decay = cfg.weight_decay)
    }[cfg.optimizer_type]

    #---------------------------------#
    # 开始模型训练
    #---------------------------------#
    UnFreeze_flag = False    # 临时变量，标识冻结阶段结束，以初始化一次解冻阶段的参数
    loss_history = LossHistory(model,cfg)
    model_train = torch.nn.DataParallel(model)
    model_train = model.to(cfg.device)
    if cfg.Freeze_Train:
        model.freeze_vgg()

    for epoch in range(cfg.Init_Epoch,cfg.UnFreeze_Epoch):

        if epoch >= cfg.Freeze_Epoch and not UnFreeze_flag and cfg.Freeze_Train:
            # 开始解冻训练
            UnFreeze_flag = True
            model.unfreeze_vgg()
            batch_size,Init_lr_fit,lr_scheduler_func , epoch_step , epoch_step_val = init_lr_epoch(cfg,False,num_train,num_val)

            trainSet_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers= 1,pin_memory=True,
                                        drop_last=True,collate_fn=unet_dataset_collate)
            valSet_loader   = DataLoader(val_dataset,batch_size=batch_size,shuffle=True,num_workers=1,pin_memory=True,
                                        drop_last=True,collate_fn=unet_dataset_collate)


        # 设置优化器的学习率衰减策略
        set_optimizer_lr(optimizer,lr_scheduler_func,epoch)
        # 进行一轮训练
        Train_OneEpoch(cfg,model_train,model,optimizer,loss_history,epoch,epoch_step,epoch_step_val,trainSet_loader,valSet_loader)

    loss_history.writer.close()





    