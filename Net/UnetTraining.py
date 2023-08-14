#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
:File       :UnetTraining.py
:Description:本文件提供了以下功能
    1.CE Loss、Dice Loss、Focal Loss
    2.学习率调度函数,用于在训练过程中动态调整学习率
    3.在冻结训练策略中对于冻结阶段和解冻阶段下的epoch数和初始的学习率需重新计算,所以就封装成一个函数
    4.训练1个epoch的函数,包含具体训练细节
    5.定义LossHistory类来记录和可视化训练过程中的损失值和验证损失值,以便进行评估和分析模型的性能
    6.自定义函数用在DataLoader中collate_fn使用
:EditTime   :2023/08/05 14:43:52
:Author     :Kiumb
'''
import os 
import math 
import torch
import numpy as np
import scipy.signal
import torch.nn as nn
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
import torch.nn.functional as F 
from Net.UnetMetric import f_score
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler as GradScaler          # 使用混精度来加速计算
from Utils.Utils import get_lr

def CE_Loss(inputs,targets,cls_weight,num_classes):
    '''
    :Description:交叉熵损失函数
    :Parameter  inputs:UNet训练结果 ([mini_batch,num_classes,height,width], raw data )
                targets:标签
                cls_weight:每一类的权重
                num_clsses:语义分割的总类别数
    :Return     CE_loss:交叉熵损失值
    '''
    n,c,h,w = inputs.size()
    nt,ht,wt= targets.size()

    if w != wt and h != ht:
        # 网络输出与标签尺寸不一致
        # align_corners设置为True,保证在进行插值时候,会将输入和输出张量的角点像素的值完全保留
        inputs = F.interpolate(inputs,(ht,wt),mode='bilinear',align_corners=True)
    
    # dim - [mini_batch,height,width,num_classes] -> [mini_batch*height*width,num_classes] 
    # contiguous方法一般用于transpose后和view前,返回了一个内存连续的有相同数据的tensor
    # transpose后的变量是原来变量的浅拷贝,而view会在原本的变量上变形。而contiguous相当于返回了tensor的深拷贝
    inputs_tmp = inputs.transpose(1,2).transpose(2,3).contiguous().view(-1,c)
    targets_tmp= targets.view(-1)

    CE_loss   = nn.CrossEntropyLoss(weight=cls_weight,ignore_index=num_classes)(inputs_tmp,targets_tmp)

    return CE_loss

def Dice_Loss(inputs,targets,beta=1,smooth=1e-5):
    '''
    :Description:dice loss 对正负样本严重不平衡的场景有着不错的性能,
        训练过程中更侧重对前景区域的挖掘。但训练loss容易不稳定,尤其是小目标的情况下。
    :Parameter  inputs:UNet训练结果
                targets:对应标签
                beta&smooth:dice loss 公式参数
    :Return     Dice_loss:损失值
    '''
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = targets.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = targets.view(n, -1, ct)

    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    Dice_loss = 1 - torch.mean(score)
    return Dice_loss
    
def Focal_Loss(inputs, targets, cls_weights, num_classes, alpha=0.5, gamma=2):
    '''
    :Description:用于图像领域解决数据不平衡造成的模型性能问题
    :Parameter  inputs:UNET训练结果
                targets:对应标签
                cls_weight:每一类的权重
                num_classes:语义分割的总类别数
                alpha&gamma:Focal loss公式参数
    :Return     Focal_loss:损失值
    '''
    n, c, h, w = inputs.size()
    nt, ht, wt = targets.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = targets.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    Focal_loss = loss.mean()
    return Focal_loss

def _lr_scheduler(lr_decay_type,lr,min_lr,total_iters,warmup_iters_ratio = .05,warmup_lr_ratio = .1,
                  no_aug_iter_ratio=.05,step_num=10):
    '''
    :Description:根据lr_decay_type来选择学习率衰减策略
    :Parameter  lr_decay_type:'cos' or 'step'
                lr:当前学习率
                min_lr&total_iters&warmup_iters_ratio&warmup_lr_ratio&no_aug_iter_ratio&step_num:cos和step函数所需参数
    :Return     func:返回lr_decay_type指定的衰减策略函数
    '''
    def yolox_warm_cos_lr(lr,min_lr,total_iters,warmup_total_iters,warmup_lr_start,no_aug_iter,iters):
        if iters <= warmup_total_iters:
            # 在热身阶段，学习率逐渐增加
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters),2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            # 在无增强阶段，学习率固定为最小学习率
            lr = min_lr
        else:
            # 在余下的阶段，学习率根据余弦函数进行衰减
            lr = min_lr + 0.5 * (lr-min_lr) * (
                1.0 + math.cos(math.pi*(iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )

        return lr 
    def step_lr(lr,decay_rate,step_size,iters):
        # 根据步长衰减策略计算学习率
        # 根据迭代次数和预定义的步长，它逐步减少学习率
        if step_size < 1:
            raise ValueError("step_size must above 1")
        n      = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == 'cos':
        # 选择cos衰减策略
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters,1),3)
        warmup_lr_start    = max(warmup_lr_ratio*lr,1e-6)
        no_aug_iter        = min(max(no_aug_iter_ratio*total_iters,1),15)

        func = partial(yolox_warm_cos_lr,lr,min_lr,total_iters,warmup_total_iters,warmup_lr_start,no_aug_iter)
    
    else:
        # 选择step衰减策略
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size  = total_iters / step_num
        func = partial(step_lr,lr,decay_rate,step_size)
    return func

def set_optimizer_lr(optimizer,lr_scheduler_func,epoch):
    '''
    :Description:设置优化器的学习率
    :Parameter  optimizer:选择的优化器
                lr_scheduler_func:选择的学习率策略函数
                epoch:当前轮次
    :Return     None:
    '''
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_lr_epoch(cfg,bt_Freeze,num_train,num_val):
    '''
    :Description:
        在冻结训练策略中对于冻结阶段和解冻阶段下的epoch数和初始的学习率会重新计算,所以就封装成一个函数
    :Parameter  cfg:配置的自定义参数类
                bt_Freeze:用于判断所处的阶段,True-冻结阶段;False-解冻阶段
                num_train:测试集数量
                num_val:验证集数量
    :Return     batch_size:批大小
                Init_lr_fit:初始化学习率
                lr_scheduler_func:学习率调度函数
                epoch_step:训练集的周期数
                epoch_step_val:验证集的周期数
    '''

    batch_size = cfg.Freeze_batch_size if bt_Freeze else cfg.Unfreeze_batch_size

    # 计算初始学习率
    nbs             = 16
    lr_limit_max    = 1e-4 if cfg.optimizer_type == 'adam' else 1e-1
    lr_limit_min    = 1e-4 if cfg.optimizer_type == 'adam' else 5e-4
    Init_lr_fit     = min(max(batch_size / nbs * cfg.Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * cfg.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2) 

    lr_scheduler_func = _lr_scheduler(cfg.lr_decay_type, Init_lr_fit, Min_lr_fit, cfg.UnFreeze_Epoch) 

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
    return batch_size,Init_lr_fit,lr_scheduler_func,epoch_step,epoch_step_val   
    

def Train_OneEpoch(cfg,model_train,model,optimizer,loss_history,epoch,epoch_step,
                   epoch_step_val,trainSet_loader,valSet_loader):
    '''
    :Description:在1个epoch中训练的具体细节
    :Parameter  cfg:配置的自定义参数类
                model_train:开启了train模式的网络模型
                model:网络模型
                optimizer:优化器
                loss_history:损失值记录和可视化类
                epoch:当前的epoch
                epoch_step:训练集下的单个epoch中的迭代数
                epoch_step_val:验证集下的单个epoch中的迭代数
                trainSet_loader:训练集数据的加载器
                valSet_loader:验证集的数据加载器
    :Return     None:
    '''
    total_loss    = 0
    total_f_score = 0

    val_loss      = 0
    val_f_score   = 0
    #---------------------------------#
    # 开始训练
    #---------------------------------#
    print('Start Train')
    # 手动更新进度条
    pbar = tqdm(total = epoch_step,desc=f'Epoch {epoch + 1}/{cfg.UnFreeze_Epoch}',postfix=dict)
    # 混精度fp16加速运算
    scaler = GradScaler()
    model_train.train()
    for iteration,batch in enumerate(trainSet_loader):
        if iteration >= epoch_step:
            break
        imgs,pngs,labels = batch
        with torch.no_grad():
            weights      = torch.from_numpy(cfg.cls_weights).to(cfg.device)
            imgs         = imgs.to(cfg.device)
            pngs         = pngs.to(cfg.device)
            labels       = labels.to(cfg.device)

        optimizer.zero_grad()
        with autocast():
            # 前向传播
            outputs = model_train(imgs)
            # 损失计算
            loss = CE_Loss(outputs,pngs,weights,num_classes = cfg.num_classes)
            loss += Dice_Loss(outputs,labels)
            with torch.no_grad():
                _f_score = f_score(outputs,labels)
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        # 手动更新进度条
        pbar.set_postfix(**{
            'total loss': total_loss / (iteration + 1),
            'f_score'   : total_f_score / (iteration + 1),
            'lr'        : get_lr(optimizer)
        })
        pbar.update(1)
    pbar.close()
    print('Finish Train')

    #---------------------------------#
    # 开始验证
    #---------------------------------#
    print('Start Validation')
    pbar = tqdm(total=epoch_step_val,desc=f'Epoch {epoch + 1}/{cfg.UnFreeze_Epoch}',postfix=dict)

    model_train.train()
    for iteration,batch in enumerate(valSet_loader):
        if iteration >= epoch_step_val:
            break
        imgs,pngs,labels = batch 
        with torch.no_grad():
            weights      = torch.from_numpy(cfg.cls_weights).to(cfg.device)
            imgs         = imgs.to(cfg.device)
            pngs         = pngs.to(cfg.device)
            labels       = labels.to(cfg.device)  

            # 前向传播
            outputs = model_train(imgs)
            # 损失计算
            loss = CE_Loss(outputs,pngs,weights,num_classes = cfg.num_classes)
            loss += Dice_Loss(outputs,labels)
            _f_score = f_score(outputs,labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            pbar.set_postfix(**{
                    'val_loss'  : val_loss / (iteration + 1),
                    'f_score'   : val_f_score / (iteration + 1),
                    'lr'        : get_lr(optimizer)
            })
            pbar.update(1)
    
    pbar.close()
    print('Finish Validation')

    #---------------------------------#
    # 记录和可视化损失值和验证损失值
    #---------------------------------#
    loss_history.append_loss(epoch+1,total_loss/epoch_step,val_loss/epoch_step_val,)
    print('Epoch:'+str(epoch + 1)+'/' + str(cfg.UnFreeze_Epoch))
    print('Total Loss:%.3f || Val Loss:%.3f' % (total_loss / epoch_step,val_loss / epoch_step_val))

    #---------------------------------#
    # 保存权值
    #---------------------------------#
    if (epoch + 1) % cfg.save_period == 0 or epoch + 1 == cfg.UnFreeze_Epoch:
        torch.save(model.state_dict(), os.path.join(cfg.save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

    if len(loss_history.val_losses) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_losses):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_epoch_weights.pth"))
        
    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "last_epoch_weights.pth"))

class LossHistory():
    '''
    :Description:记录和可视化训练过程中的损失值和验证损失值,以便进行评估和分析模型的性能
    :Args  log_dir:过程文件(记录损失值和验证损失值的文件、绘制的趋势图)保存路径
           model:网络结构
           input_shape:喂入网络的规定尺寸,此处是为了展示计算图
           val_loss_flag:记录验证损失值的标志位
    '''
    def __init__(self,model,cfg,val_loss_flag = True) -> None:
        self.log_dir = cfg.log_dir
        self.val_loss_flag = val_loss_flag
        
        self.losses = []
        if self.val_loss_flag:
            self.val_losses = []
        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            # tensorboard绘制计算图
            # 随机构造[batchsize,channel,height,width]的输入
            dummy_input = torch.randn(2,3,cfg.input_shape[0],cfg.input_shape[1])
            self.writer.add_graph(model,dummy_input)
        except:
            pass
    
    def append_loss(self,epoch,loss,val_loss = None):
        '''
        :Description:实现了以下两个功能
            1.将每个epoch的损失值进行保存到本地文件中
            2.并将每个epoch的损失值绘制到tensorboard上
        :Parameter  epoch:当前的epoch
                    loss:当前的loss
                    val_loss:当前的验证loss
        :Return     None:
        '''
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_losses.append(val_loss)
        with open(os.path.join(self.log_dir,'epoch_loss.txt'),'a') as f:
            f.write(str(loss))
            f.write('\n')
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir,'epoch_val_loss.txt'),'a') as f:
                f.write(str(val_loss))
                f.write('\n')

        self.writer.add_scalar('loss',loss,epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss',val_loss,epoch)
        
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters,self.losses,'red',linewidth = 2,label = 'train loss')
        if self.val_loss_flag:
            plt.plot(iters,self.val_losses,'coral',linewidth = 2,label = 'val loss')

        try: 
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters,scipy.signal.savgol_filter(self.losses,num,3),'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass
    
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")



def unet_dataset_collate(batch):
    '''
    :Description:DataLoader中collate_fn使用
    :Parameter  batch:一个批数据
    :Return     images:喂入网络数据
                pngs:标签数据
                seg_labels:独热形式的标签数据
    '''
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    # 将列表转换成tensor类型
    # 需转换成tensor类，否则没有to、cuda等方法
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels