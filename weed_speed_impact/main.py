#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/12/28 19:20
# @Author : FengSong
import math
import os
import random
import warnings
import numpy as np
import torch

from data.dataset import DataSet
from weed_speed_impact.feature_extraction import ResNet, Bottleneck
from weed_speed_impact.utils.tensor_utils import TensorUtils
from weed_speed_impact.cnn import ModelCNN

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    seed = 10
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # # 读取训练数据
    # train_data = DataSet.read_data('./experimental_data/data_train.csv')
    # # 过滤数据
    # train_data = DataSet.filter_data(train_data, threshold=0.1, shift=2)
    # train_data = train_data.values.astype(np.float32)
    # train_data = torch.tensor(train_data)
    # torch.save(train_data, './experimental_data/data_train_5000.pt')
    print("==================================读取训练数据===============p=====================")
    train_data = TensorUtils.load('./experimental_data/data_train_5000.pt')
    # 将数据分为特征和标签
    train_data_value, train_data_label = TensorUtils.separate_data(train_data)
    print("==================================提取训练数据特征====================================")
    resnet = ResNet(block=Bottleneck, block_num=[3, 4, 6, 3], num_classes=1022)
    print(resnet)
    train_x = resnet.extract(train_data_value)
    # 处理标签数据
    train_y = TensorUtils.handle_data_label(train_data_label)
    print("==================================读取验证数据====================================")
    valid_data = TensorUtils.load('./experimental_data/data_valid.pt')
    # 将数据分为特征和标签
    valid_data_value, valid_data_label = TensorUtils.separate_data(valid_data)
    print("==================================提取验证数据特征====================================")
    valid_x = resnet.extract(valid_data_value)
    # 处理标签数据
    valid_y = TensorUtils.handle_data_label(valid_data_label)
    print("==================================开始训练====================================")
    # 创建ModelCNN对象
    model = ModelCNN()
    print(model)
    # # 训练模型
    model_path = './model/cnn.pt'
    model_best_path = './model/cnn_best.pt'
    # model.train_model(train_x, train_y, valid_x, valid_y, model_path, model_best_path)

    # 读取测试数据
    test_data = TensorUtils.load('./experimental_data/data_test.pt')
    # 将测试数据分为特征和标签
    test_data_value, test_data_label = TensorUtils.separate_data(test_data)
    # 提取融合特征
    test_x = resnet.extract(test_data_value)
    # 处理标签数据
    test_y = TensorUtils.handle_data_label(test_data_label)
    print("==================================测试数据====================================")
    model = ModelCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    outputs = model.predict(test_x)
    # 训练集混淆矩阵
    model.score(outputs.detach().numpy(), test_y.detach().numpy())
