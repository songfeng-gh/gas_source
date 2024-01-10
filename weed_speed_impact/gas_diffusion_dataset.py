#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/1/2 13:58
# @Author : FengSong
from torch.utils.data import Dataset


class GasDiffusionDataset(Dataset):
    # 构造函数
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # 返回数据集大小
    def __len__(self):
        return self.x.size(0)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.x[index], self.y[index]
