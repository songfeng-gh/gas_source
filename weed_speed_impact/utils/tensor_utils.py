#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/1/3 11:13
# @Author : FengSong
from time import time

import torch


class TensorUtils:
    @staticmethod
    def load(path):
        start = time()
        data = torch.load(path)
        end = time()
        print("读取数据时间为：" + str(end - start) + "秒")
        return data

    @staticmethod
    def separate_data(data):
        data_value = data[:, 5:]
        data_label = data[:, 1:5]
        return data_value, data_label

    @staticmethod
    def handle_data_label(data_label):
        temp = data_label.clone()
        temp = temp / 3
        temp = torch.tensor(temp, dtype=torch.int32)
        res = torch.zeros(len(data_label), 500)
        for i, x, y, z in zip(range(0, len(data_label)), temp[:, 0], temp[:, 1], temp[:, 2]):
            index = 100 * z + 10 * y + x
            res[i][index] = 1
        return res
