# -*- coding: utf-8 -*-
# @Time : 2023/6/2 20:05
# @Author: LZ
import csv
import torch
import joblib
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pssm_onehot import TrainDataset
from pssm_onehot import TestDataset
from sklearn.neighbors import KNeighborsClassifier
import datetime

# train的大致流程
batch_size = 8 #  [40,20,10]
train_max_length = 105
test_max_length = 150
test_num_samples = 71 # 2459
train_num_samples = 5856
test_drop_last = test_num_samples % batch_size != 0
train_drop_last = train_num_samples % batch_size != 0
# Learning_rate = 0.0001   # [0.01,0.001,0.0001]
# 训练轮数
epoch = 100  #  [200,400,800]

print(datetime.datetime.now())

# 读取序列列表
sequence_list = []
label_list = []
with open('dataset/train/GPU/5856/101-105.csv', 'r') as sequence_file:
# with open('dataset/train/GPU/8059/train.csv', 'r') as sequence_file:
# with open('dataset/train/test/seq_lab/seq_train.csv', 'r') as sequence_file:
    reader = csv.reader(sequence_file)
    next(reader)
    for row in reader:
        sequence = row[1]
        label = row[2]  # 假设二级结构标签在CSV文件的第二列
        sequence_list.append(sequence)#将信息添加进列表中
        label_list.append(label)

# 设置PSSM文件夹路径
pssm_folder = 'dataset/train/GPU/5856/pssm'
# pssm_folder = 'dataset/train/GPU/8059/pssm'
# pssm_folder = 'dataset/train/test/pssm'

# 定义训练设备
# device = torch.device("cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建数据集对象
train_dataset = TrainDataset(sequence_list, pssm_folder, label_list, train_max_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=train_drop_last)  # 创建数据加载器，设置合适的批量大小和其他参数

# 测试
# 读取序列列表
sequence_list = []
label_list = []
# with open('dataset/test/GPU-test/CASP10(len100num1733)/CASP10.csv', 'r') as sequence_file:
# with open('dataset/test/GPU-test/CASP11(len410num10)/CASP11.csv', 'r') as sequence_file:
with open('dataset/test/GPU-test/CB513(len150num71)/CB513.csv', 'r') as sequence_file:
# with open('dataset/test/test/seq_lab/seq_test.csv', 'r') as sequence_file:
    reader = csv.reader(sequence_file)
    next(reader)
    for row in reader:
        sequence = row[1]
        label = row[2]  # 假设二级结构标签在CSV文件的第二列
        sequence_list.append(sequence)#将信息添加进列表中
        label_list.append(label)

# 设置PSSM文件夹路径
# pssm_folder = 'dataset/test/GPU-test/CASP10(len100num1733)/pssm_CASP10'
# pssm_folder = 'dataset/test/GPU-test/CASP11(len410num10)/pssm_CASP11'
pssm_folder = 'dataset/test/GPU-test/CB513(len150num71)/pssm_CB513'
# pssm_folder = 'dataset/test/test/pssm'

# 创建数据集对象
test_dataset = TestDataset(sequence_list, pssm_folder, label_list, test_max_length)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=test_drop_last)  # 创建数据加载器，设置合适的批量大小和其他参数

# model = SERT_StructNet()

# 选择设备
# model = model.to(device)

# # 定义损失函数
# loss_fn = torch.nn.CrossEntropyLoss()
# # 选择设备
# loss_fn = loss_fn.to(device)

# 定义优化器   Adam
# optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate, weight_decay=1e-8)

# SOV = [(∑[[(minov(S1, S2) + σ(S1, S2))*length(S1)] / maxov(S1, S2)]]*(100 / N)
def calculate_sov(S1, S2):
    if len(S1) != len(S2):
        raise ValueError("S1 and S2 must have the same length")

    NSov = len(S1)  # 总的残基数
    length_S1 = len(S1)  # S1 的残基长度
    length_S2 = len(S2)  # S1 的残基长度

    minov = 0  # 最小重叠长度
    maxov = 0  # 最大重叠长度
    S0 = 0  # 结构相同的片段数量

    for i in range(NSov):
        if S1[i] == S2[i]:
            minov += 1
            maxov += 1
            S0 += 1
        else:
            maxov += 1

    sigma = min(maxov - minov, minov, length_S1 // 2, length_S2 // 2)  # σ(S1, S2)
    numerator = minov + sigma
    denominator = maxov
    a = numerator / denominator
    b = a * length_S1
    c = b * S0
    sov = c / NSov

    return sov

def train_knn(train_dataloader, test_dataloader):

    knn_model = KNeighborsClassifier(n_neighbors=1)  # 初始化KNN模型

    best_train_accuracy = 0.0  # 初始化最佳准确率为0
    best_test_accuracy = 0.0  # 初始化最佳准确率为0

    for i in range(epoch):
        print("--------第 {} 轮训练开始--------".format(i+1))

        # 训练开始
        train_correct_predictions = 0
        total_samples = 0
        total_sov = 0.0

        # 训练开始
        for batch_data in train_dataloader:
            encoded_fusion = batch_data[0].to("cpu")  # 将特征移到 CPU 上
            labels = batch_data[1].to("cpu")  # 将标签移到 CPU 上

            # features = encoded_fusion.view(encoded_fusion.size(0), -1).numpy()  # 转换特征为 NumPy 数组
            features = encoded_fusion.view(-1, encoded_fusion.size(-1)).numpy()
            labels = torch.argmax(labels, dim=2).numpy().flatten()

            # 在每个批次内训练一个 KNN 模型
            knn_model.fit(features, labels)

            # 在当前批次上预测和计算 SOV
            for features, labels in zip(encoded_fusion, labels):
                predictions = knn_model.predict(features)  # 对单个样本进行预测
                train_correct_predictions += (predictions == labels).sum().item()  # 统计正确预测的数量
                S1 = "H" if labels == 0 else ("E" if labels == 1 else "C")
                S2 = "H" if predictions[0] == 0 else ("E" if predictions[0] == 1 else "C")
                total_sov += calculate_sov(S1, S2)

            total_samples += len(encoded_fusion)

        # total_samples = batch_size * train_max_length

        # 计算每轮的平均 ACC 和 SOV
        # epoch_accuracy = correct_predictions / total_samples
        # epoch_accuracy = correct_predictions / (batch_size * train_max_length)
        epoch_accuracy = train_correct_predictions / (len(train_dataloader) * train_max_length * batch_size)
        epoch_sov = total_sov / total_samples
        # epoch_sov = total_sov / (total_samples * train_total_batches)
        # epoch_sov = total_sov / (total_samples * len(train_dataloader))
        # epoch_sov = (total_sov / train_total_batches) / train_max_length

        print("第 {} 轮的平均ACC： {:.4f}".format(i + 1, epoch_accuracy))
        print("第 {} 轮的平均SOV： {:.4f}".format(i + 1, epoch_sov))

        # 保存模型
        if epoch_accuracy > best_train_accuracy:
            best_train_accuracy = epoch_accuracy
            # 保存最优模型
            joblib.dump(knn_model, "./model_save/KNN_best_train_model.pkl")
            print("!!!!!!!!!!!!!!!!!!第 {} 轮的模型已保存!!!!!!!!!!!!!!!!!!!".format(i + 1))


        # test
        # 在测试阶段进行评估

        test_correct_predictions = 0
        test_total_samples = 0
        total_sov = 0.0

        for batch_data in test_dataloader:
            encoded_fusion = batch_data[0].to("cpu")  # 将特征移到 CPU 上
            labels = batch_data[1].to("cpu")  # 将标签移到 CPU 上

            features = encoded_fusion.view(-1, encoded_fusion.size(-1)).numpy()
            labels = torch.argmax(labels, dim=2).numpy().flatten()

            predictions = knn_model.predict(features)  # 对单个样本进行预测
            test_correct_predictions += (predictions == labels).sum().item()  # 统计正确预测的数量

            for features, labels in zip(encoded_fusion, labels):
                S1 = "H" if labels == 0 else ("E" if labels == 1 else "C")
                S2 = "H" if predictions[0] == 0 else ("E" if predictions[0] == 1 else "C")
                total_sov += calculate_sov(S1, S2)

            test_total_samples += len(encoded_fusion)

        # 计算测试集上的平均 ACC 和 SOV
        test_accuracy = test_correct_predictions / (len(test_dataloader) * test_max_length * batch_size)
        test_sov = total_sov / test_total_samples
        # test_sov = (total_sov / train_total_batches) / test_max_length

        print("测试集上的平均ACC： {:.4f}".format(test_accuracy))
        print("测试集上的平均SOV： {:.4f}".format(test_sov))

        # 保存模型
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            joblib.dump(knn_model, "./model_save/KNN_best_test_model.pkl")
            print("-----------------!!!!!!!!!!!!!!!!!!第 {} 轮的ACC最优!!!!!!!!!!!!!!!!!!!-----------------".format(i + 1))

if __name__ == '__main__':
    train_knn(train_dataloader, test_dataloader)