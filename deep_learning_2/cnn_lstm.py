import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from time import time
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics


class GasDiffusionDataset(Dataset):
    # 构造函数
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    # 返回数据集大小
    def __len__(self):
        return self.data_tensor.size(0)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


# se注意力机制
class Se(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(Se, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(None, in_channel))
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel // reduction, out_features=in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.shape[1], x.shape[2])
        out = self.fc(out)
        out = out.view(x.shape[0], x.shape[1], x.shape[2])
        return out * x


# cnn-lstm网络
class CNNLSTM(nn.Module):
    def __init__(self, input_size):
        super(CNNLSTM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, 100, kernel_size=3, stride=1),
            nn.BatchNorm2d(100),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(100, 50, kernel_size=(148, 98), stride=1),
            nn.BatchNorm2d(50),
        )

        # batch_first如果是True，则input为(batch, seq, input_size)。默认值为：False（seq_len, batch, input_size）
        self.LSTM = nn.LSTM(input_size=50, hidden_size=150, num_layers=1, batch_first=False)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(150, 500),
        )

        self.dropout = nn.Dropout(p=0.5)

        self.threshold = 0.5
        self.se_cnn = Se(50)
        self.se_lstm = Se(150)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        batch_size, channel, height, wide = x.shape
        x = x.view(-1, batch_size, channel)
        x = self.se_cnn(x)
        x, h = self.LSTM(x)
        x = self.dropout(x)
        x = self.se_lstm(x)
        x = x[-1]
        x = self.dense(x)

    # loss_rate kl和BCELoss的比例
    # verify_rate 训练集和验证集的比例
    def train_model(self, train_x, train_y, loss_rate=0.2, verify_rate=0.1, model_path="./model/model.pt",
                    best_path="./model/best_model.pt"):
        train_x = train_x.reshape(-1, 2, 150, 100)
        # 获取训练集 验证集比例
        index = int((1 - verify_rate) * len(train_x))
        index = math.floor(index / 100) * 100
        # 分割训练集 验证集
        trainX = train_x[0:index, :]
        trainY = train_y[0:index, :]
        verifyX = train_x[index:, :]
        verifyY = train_y[index:, :]

        print('trainX:', len(trainX), "verifyX", len(verifyX))

        # 转为tensor
        trainX = trainX.astype(np.float32)
        trainX = torch.tensor(trainX)

        trainY = trainY.astype(np.float32)
        trainY = torch.tensor(trainY)

        verifyX = verifyX.astype(np.float32)
        verifyX = torch.tensor(verifyX)

        verifyY = verifyY.astype(np.float32)
        verifyY = torch.tensor(verifyY)

        # 创建gpu
        n_gpu = 1
        device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")
        print(torch.cuda.get_device_name(0))
        # 打印模型
        self = self.to(device)
        print(self)
        # 将训练集 验证集数据放到gpu中
        if torch.cuda.is_available():
            trainX = trainX.cuda()
            trainY = trainY.cuda()
            verifyX = verifyX.cuda()
            verifyY = verifyY.cuda()

        num_epochs = 3000  # 总训练轮数
        learning_rate = 0.01  # 初始学习率
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)  # 设定优化器
        scheduler = StepLR(optimizer, step_size=600, gamma=0.997)
        criterion_bce = torch.nn.BCELoss()  # BCELoss error for regression
        # criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')  # KL error for regression
        criterion_kl = torch.nn.KLDivLoss()  # 设置kl散度loss

        # 将训练集 验证集集成为DataLoader
        gas_diffusion_dataset = GasDiffusionDataset(trainX, trainY)
        gas_diffusion_dataset_verify = GasDiffusionDataset(verifyX, verifyY)
        train_loader = DataLoader(dataset=gas_diffusion_dataset,  # 传入的数据集, 必须参数
                                  batch_size=100,  # 输出的batch大小
                                  shuffle=True,  # 数据是否打乱
                                  num_workers=0)  # 进程数, 0表示只有主进程

        verify_loader = DataLoader(dataset=gas_diffusion_dataset_verify,  # 传入的数据集, 必须参数
                                   batch_size=100,  # 输出的batch大小
                                   shuffle=True,  # 数据是否打乱
                                   num_workers=0)  # 进程数, 0表示只有主进程

        # Train the model
        start = time()
        for epoch in range(num_epochs):
            for step, (b_x, b_y) in enumerate(train_loader):
                optimizer.zero_grad()  # clear gradients for this training step
                outputs = self(b_x)

                # zero = torch.zeros_like(outputs)
                # one = torch.ones_like(outputs)
                # pred = torch.where(outputs > threshold, one, zero)

                # obtain the loss function
                loss_bce = criterion_bce(outputs, b_y)
                # x_log = F.log_softmax(outputs, dim=1)
                # loss_kl = criterion_kl(x_log, b_y)
                loss_kl = criterion_kl(outputs, b_y)
                loss = (loss_rate * loss_kl + (1 - loss_rate) * loss_bce) / 10

                outputs = outputs.detach().cpu()
                loss.requires_grad_(True)
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

            if epoch % 10 == 0:
                print("==============================loss==================================")
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
            scheduler.step()
        end = time()
        print("训练时间为：" + str(end - start) + "秒")

        # 保存模型
        torch.save(self.state_dict(), model_path)

    def test_model(self, y_test, predicts):
        # accuracy_score（准确率得分）是模型分类正确的数据除以样本总数
        accuracy = metrics.accuracy_score(y_test, predicts)
        # 召回率（Recall）又被称为查全率，表示预测结果为正样本中实际正样本数量占全样本中正样本的比例。
        recall = metrics.recall_score(y_test, predicts, average="micro")
        # 精确率（Precision）又叫查准率，表示预测结果为正例的样本中实际为正样本的比例。
        precision = metrics.precision_score(y_test, predicts, average="micro")
        # F1 score是精确率和召回率的一个加权平均。
        F1 = metrics.f1_score(y_test, predicts, average="micro")
        return accuracy, recall, precision, F1
