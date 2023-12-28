import math

import numpy as np
import pandas as pd
import chama
from utils.Utils import Utils
from time import time
import torch
import matplotlib.pyplot as plt


class DataSet():
    @staticmethod
    def diffusion_trend():
        # 定义受体网络
        interval = 21
        x_grid = np.linspace(0, 50, interval)
        y_grid = np.linspace(0, 50, interval)
        z_grid = np.linspace(0, 15, interval)
        grid = chama.simulation.Grid(x_grid, y_grid, z_grid)
        # x_grid = np.array([*range(0, 30, 3)]) + 0.5
        # y_grid = np.array([*range(0, 30, 3)]) + 0.5
        # z_grid = np.array([5, 10])
        # grid = chama.simulation.Grid(x_grid, y_grid, z_grid)

        # 定义源
        source = chama.simulation.Source(22.04, 19.25, 6.21, 10)

        # 定义大气条件
        atm = pd.DataFrame({'Wind Direction': [130, 130],
                            'Wind Speed': [1.2, 1],
                            'Stability Class': ['A', 'A']},
                           index=[0, 10])

        # # 高斯烟羽模型
        # gauss_plume = chama.simulation.GaussianPlume(grid, source, atm)
        # gauss_plume.run()
        # signal = gauss_plume.conc
        #
        # # 二维扩散趋势图
        # signal_2d = signal[10::interval]
        # signal_2d = signal_2d.loc[signal_2d["T"] == 10]
        # Utils.sketch_diffusion_trend(signal_2d["X"].values.reshape(interval,interval),
        #                              signal_2d["Y"].values.reshape(interval,interval),
        #                              signal_2d["S"].values.reshape(interval,interval),
        #                              "test_plume")

        # 高斯烟团模型
        gauss_puff = chama.simulation.GaussianPuff(grid, source, atm, tpuff=1, tend=75, tstep=0.5)
        gauss_puff.run(grid, 0.5)
        signal = gauss_puff.conc
        signal[signal < 1e-9] = 0

        # 二维扩散趋势图
        signal_2d = signal.loc[signal["T"] == 75]
        signal_2d = signal_2d[15::21]
        interval = 21
        Utils.sketch_diffusion_trend(signal_2d["X"].values.reshape(interval, interval),
                                     signal_2d["Y"].values.reshape(interval, interval),
                                     signal_2d["S"].values.reshape(interval, interval),
                                     "test_puff")

    @staticmethod
    # 气体监测信息转换为待使用的格式
    def process_data(signal):
        res = []
        for i in np.linspace(0.5, 75, 150):
            signal_2d = signal.loc[signal["T"] == i]
            signal_five = signal_2d[0::2]
            signal_ten = signal_2d[1::2]

            concentration_five = signal_five["S"].to_numpy()
            concentration_ten = signal_ten["S"].to_numpy()

            concentration = np.concatenate((concentration_five, concentration_ten), axis=0).reshape(20, 10)

            res.append(concentration)
        res = np.array(res).reshape(150, 20, 10)
        return res

    @staticmethod
    def get_single_data(source, atm):
        # 定义受体网络
        x_grid = np.array([*range(0, 30, 3)]) + 0.5
        y_grid = np.array([*range(0, 30, 3)]) + 0.5
        z_grid = np.array([5, 10])
        grid = chama.simulation.Grid(x_grid, y_grid, z_grid)

        # 监测tend秒 每t_step秒返回一次数据
        gauss_puff = chama.simulation.GaussianPuff(grid, source, atm, tpuff=1, tend=75, tstep=0.5)
        gauss_puff.run(grid, 0.5)
        signal = gauss_puff.conc

        signal[signal < 1e-9] = 0

        res = DataSet.process_data(signal)
        return res

    @staticmethod
    def produce_data(numbers, path):
        data = []
        np.random.seed(100)
        upper = np.array([30, 30, 5, 10, 2.9, 360])
        lower = np.array([0, 0, 0, 0, 0, 0])
        label = np.random.random((numbers, 6)) * (upper - lower) + lower

        for i in range(0, len(label)):
            print(f"生成第{i + 1}条数据", i + 1)

            # 定义源
            source = chama.simulation.Source(label[i][0], label[i][1], label[i][2], label[i][3])
            # 定义大气条件
            atm = pd.DataFrame({'Wind Direction': [label[i][5]],
                                'Wind Speed': [label[i][4]],
                                'Stability Class': ['A']},
                               index=[0])
            # 获取扩散数据
            temp = DataSet.get_single_data(source, atm)
            data.append(temp)

        data = np.array(data).reshape(numbers, -1)
        res = np.concatenate((label, data), axis=1)
        res = pd.DataFrame(res)

        res.to_csv(path_or_buf=path, index_label=None)

    @staticmethod
    def read_single_csv(input_path):
        df_chunk = pd.read_csv(input_path, chunksize=2000)
        res_chunk = []
        for chunk in df_chunk:
            res_chunk.append(chunk)
        res_df = pd.concat(res_chunk)
        return res_df

    @staticmethod
    def read_data(path):
        # 读取数据
        start = time()
        # data = pd.read_csv('../实验数据/data.csv')
        # data = DataSet.read_single_csv('../实验数据/data.csv')
        data = DataSet.read_single_csv(path)
        end = time()
        print("读取数据时间为：" + str(end - start) + "秒")

        # 删除全为0的值
        data = data.drop(index=data[(data == 0).all(axis=1)].index)
        return data

    @staticmethod
    def separate_data(data, shift=0):
        data_value = data.iloc[:, 5 + shift:]
        data_label = data.iloc[:, 0:5]
        return data_value, data_label

    @staticmethod
    def split_data(data, threshold=0.7, shift=0):
        data_value, data_label = DataSet.separate_data(data, shift=shift)

        index = int(threshold * len(data))

        train_data_value = data_value.iloc[0:index]
        test_data_value = data_value.iloc[index:]
        train_data_label = data_label.iloc[0:index]
        test_data_label = data_label.iloc[index:]

        return train_data_value, test_data_value, train_data_label, test_data_label

    @staticmethod
    def split_train_verify_data(data_value, data_label, verify_rate=0.1):
        # 获取训练集 验证集比例
        index = int((1 - verify_rate) * len(data_value))
        index = math.floor(index / 100) * 100
        # 分割训练集 验证集
        train_data_value = data_value[0:index, :]
        train_data_label = data_label[0:index, :]
        verify_data_value = data_value[index:, :]
        verify_data_label = data_label[index:, :]

        return train_data_value, verify_data_value, train_data_label, verify_data_label

    @staticmethod
    def handle_data_value(data_value):
        data_value = data_value.values
        return data_value.reshape(len(data_value), 150, 20, 10)

    @staticmethod
    def handle_data_label(data_label):
        temp = data_label.iloc[:, 1:4]
        temp = temp / 3
        temp = temp.astype(int)
        res = np.zeros((len(data_label), 500))
        for i, x, y, z in zip(range(0, len(data_label)), temp.iloc[:, 0], temp.iloc[:, 1], temp.iloc[:, 2]):
            index = 100 * z + 10 * y + x
            res[i][index] = 1
        return res

    @staticmethod
    def filter_data(data, threshold=0.1, shift=0):
        # 将所有sum<threshold的数据删除
        data_sum = data.iloc[:, 5 + shift:].apply(lambda x: x.sum(), axis=1)
        data = data.drop(data_sum[data_sum < threshold].index)
        return data

    @staticmethod
    def numpy_to_tensor(data):
        data = data.astype(np.float32)
        data = torch.tensor(data)
        return data

    @staticmethod
    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    @staticmethod
    def standardization(data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma


if __name__ == '__main__':
    # DataSet.diffusion_trend()
    # 产生数据
    data_path = '../实验数据/data_train.csv'
    DataSet.produce_data(8000, data_path)
    # 获取训练数据
    train_data = DataSet.read_data(data_path)
    train_data = DataSet.filter_data(train_data, shift=2)
    train_data_value, verify_data_value, train_data_label, verify_data_label = DataSet.split_data(train_data,
                                                                                                  threshold=0.9,
                                                                                                  shift=0)
    print(train_data.shape)

    # # 将数据转为特定格式
    # train_x = DataSet.handle_data_value(train_data_value)
    # test_x = DataSet.handle_data_value(verify_data_value)
    # train_y = DataSet.handle_data_label(train_data_label)
    # test_y = DataSet.handle_data_label(verify_data_label)
    # # 获取测试数据
    # test_data = DataSet.read_data('../实验数据/data_test.csv')
    # test_data_value, test_data_label = DataSet.separate_data(test_data, shift=2)
    # print(train_data_value)
