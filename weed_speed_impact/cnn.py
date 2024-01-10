#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/1/2 11:23
# @Author : FengSong
# 一维卷积对表格类型数据进行训练
import math
from time import time
import torch
from sklearn import metrics
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from weed_speed_impact.gas_diffusion_dataset import GasDiffusionDataset
from weed_speed_impact.feature_extraction import ResNet, Bottleneck


# 两层卷积层，后面接一个全连接层
class ModelCNN(nn.Module):
    def __init__(self):
        super(ModelCNN, self).__init__()
        self.layer1 = nn.Sequential(
            # 输入通道一定为1，输出通道为卷积核的个数，3为卷积核的大小（实际为一个[1,2]大小的卷积核）
            # (input-kernel_size+2*padding)/stride+1
            nn.Conv1d(1, 16, 3, padding=1, stride=2),
            nn.BatchNorm1d(16),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, 5, padding=2, stride=4),
            nn.BatchNorm1d(32),
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 500),
        )

        self.dropout = nn.Dropout(p=0.5)
        self.threshold = 0.5

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dense(x)

        output = torch.sigmoid(x)
        return output

    def train_model(self, train_value, train_label, valid_value, valid_label, model_path="./model/model.pt",
                    best_path="./model/best_model.pt",
                    loss_rate=0.2):
        train_value = train_value.view(-1, 1, train_value.shape[1])
        valid_value = valid_value.view(-1, 1, valid_value.shape[1])

        print(f"训练集数据长度:{len(train_value)}")
        print(f"验证集数据长度:{len(valid_value)}")

        if torch.cuda.is_available():
            train_value = train_value.cuda()
            train_label = train_label.cuda()
            valid_value = valid_value.cuda()
            valid_label = valid_label.cuda()
            self = self.cuda()

        num_epochs = 1200  # 总训练轮数
        learning_rate = 0.01  # 初始学习率
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)  # 设定优化器
        scheduler = StepLR(optimizer, step_size=600, gamma=0.997)
        criterion_bce = torch.nn.BCELoss()  # BCELoss error for regression
        criterion_kl = torch.nn.KLDivLoss()  # 设置kl散度loss

        # 将训练集 验证集存为DataLoader
        gas_diffusion_dataset = GasDiffusionDataset(train_value, train_label)

        train_loader = DataLoader(dataset=gas_diffusion_dataset,  # 传入的数据集, 必须参数
                                  batch_size=100,  # 输出的batch大小
                                  shuffle=True,  # 数据是否打乱
                                  num_workers=0)  # 进程数, 0表示只有主进程

        start = time()
        for epoch in range(num_epochs):
            for step, (b_x, b_y) in enumerate(train_loader):
                optimizer.zero_grad()  # clear gradients for this training step
                outputs = self.forward(b_x)
                loss_bce = criterion_bce(outputs, b_y)
                loss_kl = criterion_kl(outputs, b_y)
                loss = loss_bce
                loss.requires_grad_(True)
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
            if epoch % 10 == 0:
                print("==============================训练集==================================")
                print("==============================train_loss==================================")
                print("Epoch: %d, train_loss: %1.5f" % (epoch, loss.item()))
                predict_train = outputs.cpu()
                # 将predict转换到0,1之间
                zero = torch.zeros_like(predict_train)
                one = torch.ones_like(predict_train)
                predict_train = torch.where(predict_train > self.threshold, one, zero)
                self.score(predict_train.detach().numpy(), b_y.cpu().detach().numpy())
                print("==============================验证集==================================")
                best_res = 1e6
                predict_valid = self.forward(valid_value)
                # 获取验证集损失
                valid_loss_bce = criterion_bce(predict_valid, valid_label)
                valid_loss_kl = criterion_kl(predict_valid, valid_label)
                valid_loss = (loss_rate * valid_loss_kl + (1 - loss_rate) * valid_loss_bce) / 10
                print("==============================valid_loss==================================")
                print("Epoch: %d, valid_loss: %1.5f" % (epoch, valid_loss.item()))
                # 将predict放到CPU
                predict_valid = predict_valid.cpu()
                # 将predict转换到0,1之间
                zero = torch.zeros_like(predict_valid)
                one = torch.ones_like(predict_valid)
                predict_valid = torch.where(predict_valid > self.threshold, one, zero)
                # 验证集混淆矩阵
                self.score(predict_valid.detach().numpy(), valid_label.cpu().detach().numpy())
                # 是否有更优模型
                if valid_loss < best_res:
                    best_res = valid_loss
                    torch.save(self.state_dict(), best_path)
            scheduler.step()
        end = time()
        print("训练时间为：" + str(end - start) + "秒")
        # 保存模型
        torch.save(self.state_dict(), model_path)

    def predict(self, test_x):
        test_x = test_x.view(-1, 1, test_x.shape[1])
        print(f"训练集数据长度:{len(test_x)}")
        # 预测
        outputs = self.forward(test_x)
        zero = torch.zeros_like(outputs)
        one = torch.ones_like(outputs)
        outputs = torch.where(outputs > 0.5, one, zero)
        return outputs

    def score(self, pred_y, real_y):
        # accuracy_score（准确率得分）是模型分类正确的数据除以样本总数
        accuracy = metrics.accuracy_score(real_y, pred_y)
        # 召回率（Recall）又被称为查全率，表示预测结果为正样本中实际正样本数量占全样本中正样本的比例。
        recall = metrics.recall_score(real_y, pred_y, average="micro")
        # 精确率（Precision）又叫查准率，表示预测结果为正例的样本中实际为正样本的比例。
        precision = metrics.precision_score(real_y, pred_y, average="micro")
        # F1 score是精确率和召回率的一个加权平均。
        F1 = metrics.f1_score(real_y, pred_y, average="micro")
        print("accuracy:", accuracy, '\n', "precision:", precision, '\n',
              "recall:", recall, '\n', "F1 :", F1)
        return accuracy, recall, precision, F1
