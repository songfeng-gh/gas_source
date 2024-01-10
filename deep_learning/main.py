import matplotlib.pyplot as plt
from data.dataset import DataSet
from cnn_lstm import CNNLSTM as cl
import numpy as np
import torch
import warnings

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # 读取数据
    data = DataSet.read_data('../实验数据/data_1000.csv')
    # 数据预处理 将所有sum<0.1的数据删除
    data_sum = data.iloc[:, 5:].apply(lambda x: x.sum(), axis=1)
    data = data.drop(data_sum[data_sum < 0.1].index)
    # 分离数据训练集、测试集
    train_data_value, test_data_value, train_data_label, test_data_label = DataSet.split_data(data)
    # 将数据转为给定格式
    train_x = DataSet.handle_data_value(train_data_value)
    test_x = DataSet.handle_data_value(test_data_value)
    train_y = DataSet.handle_data_label(train_data_label)
    test_y = DataSet.handle_data_label(test_data_label)

    # 创建模型
    model = cl(input_size=150)
    # 训练模型
    model_path = './model/cnn_lstm_opt_standardization.pt'
    model_best_path = './model/cnn_lstm_opt_cnn_lstm_opt_standardization_best.pt'
    # model.train_model(train_x, train_y, model_path=model_path, best_path=model_best_path)

    # Test the model
    # 归一化或者标准化数据
    # test_x = DataSet.standardization(test_x)
    counts_list = []
    for version in np.arange(0, 0.6, 0.6):
        version = round(version, 1)
        # model_path = f'./model/cnn_lstm_best_{version}.pt'
        # 读取模型
        model = cl(input_size=150)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        test_x = test_x.astype(np.float32)
        testX = torch.tensor(test_x)

        test_y = test_y.astype(np.float32)
        testY = torch.tensor(test_y)

        outputs = model(testX)
        zero = torch.zeros_like(outputs)
        one = torch.ones_like(outputs)

        accuracy_list, recall_list, precision_list, f1_list = [], [], [], []

        for i in np.arange(0, 1, 0.1):
            pred = torch.where(outputs > i, one, zero)

            accuracy, recall, precision, f1 = model.test_model(test_y, pred.detach().numpy())
            print(f"============================={i}======================================")
            print("accuracy:", accuracy, '\n', "precision:", precision, '\n',
                  "recall:", recall, '\n', "F1 :", f1)
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

            # 画出混淆矩阵的变化
            # plt.figure()
            # plt.plot(np.arange(0, len(accuracy_list)), accuracy_list, label='accuracy')
            # plt.plot(np.arange(0, len(precision_list)), precision_list, label='precision')
            # plt.plot(np.arange(0, len(recall_list)), recall_list, label='recall')
            # plt.plot(np.arange(0, len(f1_list)), f1_list, label='f1')
            # plt.legend()
            # plt.show()

        counts = 0
        temp = outputs.detach().numpy()
        for i in range(0, temp.shape[0]):
            pred_index = np.argmax(temp[i])
            real_index = np.argwhere(test_y[i] == 1)
            if pred_index == real_index:
                counts += 1
        print(counts / len(outputs))
        counts_list.append(counts / len(outputs))

    # plt.figure()
    # plt.plot(np.arange(0, len(counts_list)), counts_list, label='accuracy')
    # plt.show()

    # 画出cnn_lstm之后的outputs
    import collections

    # bins = np.arange(0, 1, 0.1)
    # for t in temp:
    #     plt.plot(np.arange(0, 500), t, 'b*--', alpha=0.5, linewidth=1, label='acc')  # 'bo-'表示蓝色实线，数据点实心原点标注
    #     # plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
    #     plt.legend()  # 显示上面的label
    #     plt.xlabel('time')  # x_label
    #     plt.ylabel('number')  # y_label
    #     # plt.ylim(-1,1)#仅设置y轴坐标范围
    #     plt.show()
    # indices = np.digitize([0.33,0.22,0.34,0.96], bins)
    # data_count = collections.Counter(indices)
