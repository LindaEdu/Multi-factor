# -*- coding: utf-8 -*-
# @Time : 2023/6/2 20:05
# @Author: LZ
import csv
import torch
import torch.nn as nn
import joblib
import numpy as np
from torch.utils.data import DataLoader
from pssm_onehot import TrainDataset
from pssm_onehot import TestDataset
import datetime
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import pandas as pd

# train的大致流程
batch_size = 1 #  [40,20,10]
train_max_length = 105
test_max_length = 150
test_num_samples = 71 # 2459
train_num_samples = 5856
test_drop_last = test_num_samples % batch_size != 0
train_drop_last = train_num_samples % batch_size != 0
# Learning_rate = 0.0001   # [0.01,0.001,0.0001]
# 训练轮数
epoch = 10  #  [200,400,800]

print(datetime.datetime.now())

# 读取序列列表
sequence_list = []
label_list = []
# with open('dataset/train/GPU/5856/101-105.csv', 'r') as sequence_file:
# with open('dataset/train/GPU/8059/train.csv', 'r') as sequence_file:
with open('dataset/train/test/seq_lab/seq_train.csv', 'r') as sequence_file:
    reader = csv.reader(sequence_file)
    next(reader)
    for row in reader:
        sequence = row[1]
        label = row[2]  # 假设二级结构标签在CSV文件的第二列
        sequence_list.append(sequence)#将信息添加进列表中
        label_list.append(label)

# 设置PSSM文件夹路径
# pssm_folder = 'dataset/train/GPU/5856/pssm'
# pssm_folder = 'dataset/train/GPU/8059/pssm'
pssm_folder = 'dataset/train/test/pssm'

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
# with open('dataset/test/GPU-test/CB513(len150num71)/CB513.csv', 'r') as sequence_file:
with open('dataset/test/test/seq_lab/seq_test.csv', 'r') as sequence_file:
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
# pssm_folder = 'dataset/test/GPU-test/CB513(len150num71)/pssm_CB513'
pssm_folder = 'dataset/test/test/pssm'

# 创建数据集对象
test_dataset = TestDataset(sequence_list, pssm_folder, label_list, test_max_length)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=test_drop_last)  # 创建数据加载器，设置合适的批量大小和其他参数

# model = SERT_StructNet()
# model = SVM_StructNet()

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


# def plot_feature_importance(importance, features):
#     feature_importance = np.array(importance)
#     feature_names = np.array(features)
#
#     data = {'feature_names': feature_names, 'feature_importance': feature_importance}
#     fi_df = pd.DataFrame(data)
#
#     fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
#
#     plt.figure(figsize=(10, 6))
#     plt.barh(fi_df['feature_names'], fi_df['feature_importance'])
#     plt.xlabel('Feature Importance')
#     plt.ylabel('Feature Names')
#     plt.title('Feature Importance')
#     plt.gca().invert_yaxis()
#     plt.show()


def train_random_forest(train_dataloader, test_dataloader):

    # 初始化随机森林模型
    rf_model = RandomForestClassifier(n_estimators=120, max_depth=15, oob_score=True)

    train_acc_history = []
    train_sov_history = []
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

            # 在每个批次内训练一个随机森林模型
            rf_model.fit(features, labels)

            # 在当前批次上预测和计算 SOV
            for features, labels in zip(encoded_fusion, labels):
                # predictions = rf_model.predict(features.reshape(1, -1))  # 对单个样本进行预测
                predictions = rf_model.predict(features)  # 对单个样本进行预测
                train_correct_predictions += (predictions == labels).sum().item()  # 统计正确预测的数量

                S1 = "H" if labels == 0 else ("E" if labels == 1 else "C")
                S2 = "H" if predictions[0] == 0 else ("E" if predictions[0] == 1 else "C")
                total_sov += calculate_sov(S1, S2)

            total_samples += len(encoded_fusion)

        # 计算每轮的平均 ACC 和 SOV
        epoch_accuracy = train_correct_predictions / (len(train_dataloader) * train_max_length * batch_size)
        epoch_sov = total_sov / total_samples

        train_acc_history.append(epoch_accuracy)
        train_sov_history.append(epoch_sov)

        print("第 {} 轮的平均ACC： {:.4f}".format(i + 1, epoch_accuracy))
        print("第 {} 轮的平均SOV： {:.4f}".format(i + 1, epoch_sov))

        # 保存模型
        if epoch_accuracy > best_train_accuracy:
            best_train_accuracy = epoch_accuracy
            # 保存最优模型
            joblib.dump(rf_model, "./model_save/RandomForest_best_train_model.pkl")
            print("!!!!!!!!!!!!!!!!!!第 {} 轮的模型已保存!!!!!!!!!!!!!!!!!!!".format(i + 1))

        # test
        test_correct_predictions = 0
        test_total_samples = 0
        total_sov = 0.0

        for batch_data in test_dataloader:
            encoded_fusion = batch_data[0].to("cpu")  # 将特征移到 CPU 上
            labels = batch_data[1].to("cpu")  # 将标签移到 CPU 上

            features = encoded_fusion.view(-1, encoded_fusion.size(-1)).numpy()
            labels = torch.argmax(labels, dim=2).numpy().flatten()

            # 在每个批次内预测
            predictions = rf_model.predict(features)
            test_correct_predictions += (predictions == labels).sum().item()  # 统计正确预测的数量

            # 在当前批次上预测和计算 SOV
            for features, labels in zip(encoded_fusion, labels):
                # predictions = rf_model.predict(features)  # 对单个样本进行预测
                # correct_predictions += (predictions == labels).sum().item()  # 统计正确预测的数量

                S1 = "H" if labels == 0 else ("E" if labels == 1 else "C")
                S2 = "H" if predictions[0] == 0 else ("E" if predictions[0] == 1 else "C")
                total_sov += calculate_sov(S1, S2)

            test_total_samples += len(encoded_fusion)

        # 计算测试集上的平均 ACC 和 SOV
        test_accuracy = test_correct_predictions / (len(test_dataloader) * test_max_length * batch_size)
        test_sov = total_sov / test_total_samples

        print("测试集上的平均ACC： {:.4f}".format(test_accuracy))
        print("测试集上的平均SOV： {:.4f}".format(test_sov))

        # 保存模型
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            joblib.dump(rf_model, "./model_save/RandomForest_best_test_model.pkl")
            print("-----------------!!!!!!!!!!!!!!!!!!第 {} 轮的ACC最优!!!!!!!!!!!!!!!!!!!-----------------".format(i + 1))

    # # 获取特征重要性
    # feature_importance = rf_model.feature_importances_
    # features = ['Feature' + str(i) for i in range(len(feature_importance))]
    # plot_feature_importance(feature_importance, features)

if __name__ == '__main__':
    train_random_forest(train_dataloader, test_dataloader)


# def train(model, train_dataloader, loss_fn, optimizer):
#
#     # model.train()
#     train_loss_history = []
#     train_acc_history = []
#     test_loss_history = []
#     test_acc_history = []
#     best_loss = 9999
#
#     for i in range(epoch):
#
#         print("--------第 {} 轮训练开始--------".format(i+1))
#         model.train()
#
#         correct_predictions = 0
#         train_loss_total = 0
#
#         # 在训练循环之前初始化SOV变量
#         train_total_sov = 0.0
#         train_total_batches = 0  # 记录总共有多少个 batch
#
#         # 训练开始
#         for batch_data in train_dataloader:
#
#             encoded_fusion = batch_data[0].to(device)  # 获取第一个元素，解包批量数据
#             labels = batch_data[1].to(device)
#
#             x = model(encoded_fusion)
#
#             loss = 0.0
#
#             # 获取网络输出的形状
#             batch_size, sequence_length, num_classes = x.size()
#
#             # 初始化S1和S2
#             S1 = ""
#             S2 = ""
#
#             # 循环遍历每一个时间步
#             for t in range(sequence_length):
#                 # 在每个时间步选择预测值和标签，通过索引‘t’进行对应
#                 protein_pred_t = x[:, t, :]
#                 labels_t = labels[:, t, :]
#                 # 将预测值和标签的形状调整为 (batch_size, num_classes)
#                 protein_pred_t = protein_pred_t.view(batch_size, num_classes)
#                 labels_t = labels_t.view(batch_size, num_classes)
#
#                 # 计算当前时间步的损失。protein_pred_t是一个时间步上的蛋白质特征     为匹配标签形状，使用torch.argmax()函数将labels_t转换为类别索引---找到one-hot中的最大值索引
#                 # eg:torch.argmax(labels_t, dim=1)：tensor([[1, 0, 0]])-->tensor([0])
#                 # eg:protein_pred_t为 [0.1, 0.8, 0.1], 经过argmax的labels_t为最大索引为，然后两者进行对应
#                 loss_t = loss_fn(protein_pred_t, torch.argmax(labels_t, dim=1))
#                 # 累加损失
#                 loss += loss_t
#
#                 # 计算每个时间步的准确率
#                 predicted_labels = torch.argmax(protein_pred_t, dim=1)  # 获取每个时间步预测结果中概率最大的类别索引，即预测的标签
#                 true_labels = torch.argmax(labels_t, dim=1)
#                 correct_predictions += torch.sum(predicted_labels == true_labels).item()  # 进行比较，得到布尔值（True or flase）
#
#                 # 构建S1和S2
#                 S1 += "H" if true_labels[0].item() == 0 else ("E" if true_labels[0].item() == 1 else "C")
#                 S2 += "H" if predicted_labels[0].item() == 0 else ("E" if predicted_labels[0].item() == 1 else "C")
#
#             # 计算当前批次的SOV
#             sov = calculate_sov(S1, S2)
#             train_total_sov += sov
#             train_total_batches += 1
#
#             total_predictions = sequence_length * batch_size
#
#             # 计算平均损失
#             loss /= total_predictions
#
#             # 优化器优化模型
#             # 梯度清零
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # 算总LOSS的和
#             train_loss_total += loss
#
#         print("epoch: {}".format(i+1))
#         # 计算一个epoch的平均loss
#         epoch_loss = train_loss_total.item() / len(train_dataloader)
#         train_loss_history.append(epoch_loss)
#         print("第 {} 轮的平均LOSS： {} ".format(i + 1, epoch_loss))
#
#         # 计算一个epoch的平均acc
#         epoch_accuracy = correct_predictions / (len(train_dataloader) * train_max_length * batch_size)
#         train_acc_history.append(epoch_accuracy)
#         print("第 {} 轮的平均ACC： {} ".format(i + 1, epoch_accuracy))
#
#         # 计算平均SOV
#         # SOV = [(∑[[(minov(S1, S2) + σ(S1, S2))*length(S1)] / maxov(S1, S2)]]*(100 / N)
#         # average_sov = (train_total_sov) / (len(train_dataloader) * max_length)
#         average_sov = (train_total_sov / train_total_batches) / train_max_length
#         print("第 {} 轮的平均SOV： {} ".format(i + 1, average_sov))
#
#         # 写入loss
#         with open('./log/loss.txt','a+') as f:
#             f.write(str(epoch_loss))
#             f.write('\n')
#
#         # 保存模型
#         if best_loss > epoch_loss:
#             best_loss = epoch_loss
#             torch.save(model.state_dict(), "./model_save/best_model.pkl")
#             # print("模型已保存!")
#
#         # 测试
#         model.eval()
#
#         with torch.no_grad():
#
#             correct_predictions = 0
#             test_loss_total = 0
#             test_total_sov = 0.0
#             test_total_batches = 0.0
#
#             for batch_data in test_dataloader:
#                 encoded_fusion = batch_data[0].to(device)
#                 labels = batch_data[1].to(device)
#
#                 x = model(encoded_fusion)
#                 test_loss = 0.0
#                 batch_size, sequence_length, num_classes = x.size()
#                 # 初始化S1和S2
#                 S1 = ""
#                 S2 = ""
#
#                 for t in range(sequence_length):
#
#                     protein_pred_t = x[:, t, :]
#                     labels_t = labels[:, t, :]
#                     protein_pred_t = protein_pred_t.view(batch_size, num_classes)
#                     labels_t = labels_t.view(batch_size, num_classes)
#
#                     loss_t = loss_fn(protein_pred_t, torch.argmax(labels_t, dim=1))
#                     test_loss += loss_t
#
#                     predicted_labels = torch.argmax(protein_pred_t, dim=1)  # 获取每个时间步预测结果中概率最大的类别索引，即预测的标签
#                     true_labels = torch.argmax(labels_t, dim=1)
#                     correct_predictions += torch.sum(predicted_labels == true_labels).item()  # 进行比较，得到布尔值（True or flase）
#
#                     # 构建S1和S2
#                     S1 += "H" if true_labels[0].item() == 0 else ("E" if true_labels[0].item() == 1 else "C")
#                     S2 += "H" if predicted_labels[0].item() == 0 else ("E" if predicted_labels[0].item() == 1 else "C")
#
#                 # 计算当前批次的SOV
#                 sov = calculate_sov(S1, S2)
#                 test_total_sov += sov
#                 test_total_batches += 1
#
#                 test_loss = test_loss / sequence_length
#                 # 计算总的测试LOSS
#                 test_loss_total += test_loss
#
#             test_final_loss = test_loss_total / len(test_dataloader)
#             print("第 {} 轮的测试LOSS： {} ".format(i + 1, test_final_loss))
#
#             # 计算准确率
#             test_final_accuracy = correct_predictions / (len(test_dataloader) * batch_size * test_max_length)
#             print("第 {} 轮的总测试ACC： {} ".format(i + 1, test_final_accuracy))
#
#             # 计算平均SOV
#             # SOV = [(∑[[(minov(S1, S2) + σ(S1, S2))*length(S1)] / maxov(S1, S2)]]*(100 / N)
#             # test_average_sov = (test_total_sov) / (len(train_dataloader) * max_length)
#             test_average_sov = (test_total_sov / test_total_batches) / test_max_length
#             print("第 {} 轮的总测试SOV： {} ".format(i + 1, test_average_sov))
#
#             # # 画图数据
#             # test_loss_history.append(test_final_loss)
#             # test_acc_history.append(total_accuracy)
#
#     print("--------------------------------------------------------------------------------------------------------")

# if __name__ == '__main__':
#     train(model, train_dataloader, loss_fn, optimizer)