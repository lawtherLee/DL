import torch  # PyTorch框架, 封装了张量的各种操作
from torch.utils.data import (
    TensorDataset,
)  # 数据集对象.   数据 -> Tensor -> 数据集 -> 数据加载器
from torch.utils.data import DataLoader  # 数据加载器.
import torch.nn as nn  # neural network, 封装了神经网络的各种操作
import torch.optim as optim  # 优化器
from sklearn.model_selection import train_test_split  # 训练集和测试集的划分
import matplotlib.pyplot as plt  # 绘图
import numpy as np  # 数组(矩阵)操作
import pandas as pd  # 数据处理
import time  # 时间模块


def create_dataset():
    data = pd.read_csv("./data/手机价格预测.csv")
    # print(f"data: {data.head()}")

    x, y = data.iloc[:, :-1], data.iloc[:, -1]

    x = x.astype(np.float32)
    # print(f"x: {x.head()}, {x.shape}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=3, stratify=y
    )

    train_dataset = TensorDataset(
        torch.tensor(x_train.values), torch.tensor(y_train.values)
    )

    test_dataset = TensorDataset(
        torch.tensor(x_test.values), torch.tensor(y_test.values)
    )
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))


if __name__ == "__main__":
    train_dataset, test_dataset, input_num, output_num = create_dataset()
