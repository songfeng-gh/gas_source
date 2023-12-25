import matplotlib.pyplot as plt
import math
import numpy as np


class Utils():
    # 求两点之间距离
    @staticmethod
    def dist(x, y):
        z = x - y
        return math.sqrt(np.sum(np.power(z, 2), axis=0))

    # 画出扩散趋势图
    @staticmethod
    def sketch_diffusion_trend(X, Y, res, title):
        fig = plt.figure()
        plt.xlabel("x/m", fontsize=16)
        plt.ylabel("y/m", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax = plt.subplot(1, 1, 1)
        ax.set_title(title, size=18)
        # 彩色 jet
        cs = ax.contourf(X, Y, res, cmap='Blues')
        # cs = ax.contourf(X, Y, res, cmap='jet', levels=np.linspace(0, 0.9, 7))
        cbar = fig.colorbar(cs)
        cbar.ax.tick_params(labelsize=12.5)
        cbar.ax.set_title('concentration\n(g.s-1)', fontsize=12.5)
        plt.show()

    # 生成随机浮点数
    @staticmethod
    def rand_float(high, low, amount):
        a = high - low
        b = high - a
        V = (np.random.rand(amount) * a + b).tolist()
        return V

    # 获取n个不等的随机整数
    @staticmethod
    def rand_int_unequal(low, high, num):
        R = []
        i = 0
        while i < num:
            r = int(np.random.randint(low=low, high=high, size=1))
            if r not in R:
                R.append(r)
                i += 1
        return R
