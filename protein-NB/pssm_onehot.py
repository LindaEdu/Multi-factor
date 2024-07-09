# -*- coding: utf-8 -*-
# @Time : 2023/6/4 16:11
# @Author: LZ
import torch
import csv
import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

'''
Training dataset: 80%
use one-hot encoding, encode protein sequence + PSSM  Stitching up and down  --- (len(max_len_sequence),20dimensions + 20dimensions))， 
using one-hot encoding, encode secondary structure labels---（len(label),3dimensions）
'''

# 使用PSSM和ONT-HOT，将其进行上下拼接，尺寸为（最大序列长度，2*20）
class TrainDataset:
    # 构造函数，用于初始化数据集对象，接收四个参数：蛋白质序列列表、pssm文件，二级结构标签列表、辅因子smiles化学式列表
    def __init__(self, sequence_list, pssm_folder, label_list, train_max_length):
        # 将传入的序列列表赋值给对象
        self.sequences = sequence_list
        self.pssm_folder = pssm_folder
        self.labels = label_list
        self.max_length = train_max_length
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
                            'W', 'Y']

        # 1-D趋势因子归一化+pka1,pkb2,pl4,疏水性，
        self.propertie_data = {'A': [0.90, 0.36, 0.45, 0.62, 2.34, 9.69, 6],
                               'C': [0.00, 0.00, 0.00, 0.29, 1.96, 10.28, 5.07],
                               'D': [0.36, 0.09, 0.63, -0.9, 1.88, 9.6, 3.65],
                               'E': [0.73, 0.18, 0.36, -0.74, 2.19, 9.67, 4.25],
                               'F': [0.27, 0.36, 0.18, 1.19, 1.83, 9.13, 5.48],
                               'G': [0.27, 0.27, 1.00, 0.48, 2.34, 9.6, 5.79],
                               'H': [0.09, 0.00, 0.18, -0.4, 1.82, 9.17, 7.59],
                               'I': [0.45, 0.73, 0.18, 1.38, 2.36, 9.6, 5.97],
                               'K': [1.00, 0.73, 0.45, -1.5, 2.18, 8.95, 9.74],
                               'L': [0.18, 0.00, 0.09, 1.06, 2.36, 9.6, 5.98],
                               'M': [0.18, 0.00, 0.09, 0.64, 2.28, 9.21, 5.74],
                               'N': [0.18, 0.09, 0.45, -0.78, 2.02, 8.8, 5.41],
                               'P': [0.18, 0.00, 0.73, 0.12, 1.99, 10.6, 6.3],
                               'Q': [0.36, 0.09, 0.18, -0.85, 2.17, 9.13, 5.65],
                               'R': [0.45, 0.18, 0.27, -2.53, 2.17, 9.04, 10.76],
                               'S': [0.36, 0.27, 0.55, -0.18, 2.21, 9.15, 5.68],
                               'T': [0.27, 0.45, 0.45, -0.05, 2.09, 9.1, 5.6],
                               'V': [0.45, 1.00, 0.27, 1.08, 2.32, 9.62, 5.96],
                               'W': [0.09, 0.00, 0.00, 0.81, 2.83, 9.39, 5.89],
                               'Y': [0.27, 0.27, 0.18, 0.26, 2.2, 9.11, 5.66]}

        # 将氨基酸列表中的每个氨基酸与其对应索引进行编码，生成字典amino_acid_encoding
        self.amino_acid_encoding = {acid: i for i, acid in enumerate(self.amino_acids)}
        self.label_class = ['H', 'E', 'C']
        # 将标签列表中的每个标签与其对应索引进行编码，生成字典labels_encoding
        self.labels_encoding = {lab: i for i, lab in enumerate(self.label_class)}
        # 序列最大长度
        # self.max_length = max([len(seq.replace(' ', '')) for seq in sequence_list])

    def __getitem__(self, index):
        # 根据索引获取对应位置的蛋白质序列、标签、smiles序列
        sequence = self.sequences[index]
        label = self.labels[index]
        # 调用类内方法，分别对蛋白质序列进行one-hot编码、将标签嵌入到编码序列、对smiles序列进行编码
        encoded_sequence = self.one_hot_encode_sequence(sequence)

        encoded_properties = self.encode_propertie(sequence)

        # 调用类内方法对标签编码
        encoded_label = self.one_hot_encode_label(label)
        # 读取pssm文件
        pssm_file = f"{index + 1}.csv"
        pssm_path = os.path.join(self.pssm_folder, pssm_file)
        # 调用类内方法，提取PSSM矩阵
        encoded_pssm = self.extract_pssm_data(pssm_path)
        # 左右拼接序列和Pssm
        encoded_fusion1 = torch.cat((encoded_pssm, encoded_sequence), dim=1)
        encoded_fusion = torch.cat((encoded_fusion1, encoded_properties), dim=1)
        # 返回嵌入二级结构后的蛋白质编码、smiles编码-----encoded_sequence, encoded_smiles分别传入进行特征提取
        return encoded_fusion, encoded_label

    def __len__(self):
        # 返回蛋白质序列列表长度==样本数据集数量
        return len(self.sequences)

    def one_hot_encode_sequence(self, sequence):
        encoded_sequence = []
        # 遍历蛋白质序列中的每个氨基酸，并编码成一个维度为20的列表
        for aa in sequence:
            if aa in self.amino_acid_encoding:
                encoded_aa = [0.0] * len(self.amino_acids)
                encoded_aa[self.amino_acid_encoding[aa]] = 1.0
                encoded_sequence.append(encoded_aa)
        # 填充到最大长度
        padding_length = self.max_length - len(encoded_sequence)
        encoded_sequence += [[0.0] * len(self.amino_acids)] * padding_length
        # 返回encoded_sequenc，其中存储编码后的蛋白质序列
        return torch.tensor(encoded_sequence, dtype=torch.float32)

    def encode_propertie(self, sequence):
        encoded_properties = []

        for aa in sequence:
            if aa in self.propertie_data:
                encoded_properties.append(self.propertie_data[aa])
            else:
                encoded_properties.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        padding_length = self.max_length - len(encoded_properties)
        encoded_properties += [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * padding_length

        return torch.tensor(encoded_properties, dtype=torch.float32)

    def extract_pssm_data(self, pssm_file):
        pssm_data = []
        with open(pssm_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过第一行

            # 读取文件中第3列到第22列。提取pssm信息
            for row in reader:
                selected_data = [float(value) for value in row[2:22]]
                pssm_data.append(selected_data)

        padding_length = self.max_length - len(pssm_data)
        pssm_data += [[0.0] * 20] * padding_length

        return torch.tensor(pssm_data, dtype=torch.float32)

    def one_hot_encode_label(self, label):
        encoded_label = []
        # 遍历标签序列中的每个标签，并编码成一个维度为3的列表
        for bb in label:
            if bb in self.labels_encoding:
                encoded_bb = [0] * len(self.label_class)
                encoded_bb[self.labels_encoding[bb]] = 1
                encoded_label.append(encoded_bb)

        # 填充到最大长度
        padding_length = self.max_length - len(encoded_label)
        encoded_label += [[0] * (len(self.label_class))] * padding_length

        # 返回encoded_label，其中存储编码后的标签序列
        return torch.tensor(encoded_label)

#
# train_max_length = 150
#
# # 读取序列列表
# sequence_list = []
# label_list = []
#
# # 序列、标签、smiles路径
# train_csv_file = 'dataset/train/test/seq_lab/seq_train.csv'
#
# # 设置PSSM文件夹路径
# pssm_folder = 'dataset/train/test/pssm'
#
# with open(train_csv_file, 'r') as sequence_file:
#     reader = csv.reader(sequence_file)
#     next(reader)
#     for row in reader:
#         sequence = row[1]
#         label = row[2]  # 假设二级结构标签在CSV文件的第二列
#         sequence_list.append(sequence)#将信息添加进列表中
#         label_list.append(label)
#
# # 创建数据集对象
# train_dataset = TrainDataset(sequence_list, pssm_folder, label_list, train_max_length)
#
# # 遍历数据集
# for i in range(len(train_dataset)):
#     encoded_fusion, encoded_label = train_dataset[i]
#     print(f"Sample {i + 1}:")
#     # print("Encoded Fusion:", encoded_fusion)
#     print("Encoded Fusion Shape:", encoded_fusion.shape)
#     print("lab:", encoded_label.shape)
#     # print(encoded_label)
#     # print("--------")
#     # print()

'''
Testing dataset: 20%
use one-hot encoding, encode protein sequence + PSSM  Stitching up and down  --- (len(max_len_sequence),20dimensions + 20dimensions))， 
using one-hot encoding, encode secondary structure labels---（len(label),3dimensions）,
'''


# 使用PSSM和ONT-HOT，将其进行上下拼接，尺寸为（最大序列长度，2*20）
class TestDataset:
    # 构造函数，用于初始化数据集对象，接收四个参数：蛋白质序列列表、pssm文件，二级结构标签列表、辅因子smiles化学式列表
    def __init__(self, sequence_list, pssm_folder, label_list, test_max_length):
        # 将传入的序列列表赋值给对象
        self.sequences = sequence_list
        self.pssm_folder = pssm_folder
        self.labels = label_list
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
                            'W', 'Y']

        # CB513趋势因子归一化+疏水性、pka1,pkb2,pl4,，
        self.propertie_data = {'A': [1.00, 0.36, 0.50, 0.62, 2.34, 9.69, 6],
                               'C': [0.10, 0.00, 0.08, 0.29, 1.96, 10.28, 5.07],
                               'D': [0.50, 0.09, 0.50, -0.9, 1.88, 9.6, 3.65],
                               'E': [0.90, 0.18, 0.41, -0.74, 2.19, 9.67, 4.25],
                               'F': [0.30, 0.27, 0.08, 1.19, 1.83, 9.13, 5.48],
                               'G': [0.30, 0.18, 1.00, 0.48, 2.34, 9.6, 5.79],
                               'H': [0.20, 0.00, 0.08, -0.4, 1.82, 9.17, 7.59],
                               'I': [0.40, 0.72, 0.16, 1.38, 2.36, 9.6, 5.97],
                               'K': [0.70, 0.36, 0.58, -1.5, 2.18, 8.95, 9.74],
                               'L': [1.00, 0.63, 0.33, 1.06, 2.36, 9.6, 5.98],
                               'M': [0.20, 0.00, 0.08, 0.64, 2.28, 9.21, 5.74],
                               'N': [0.30, 0.09, 0.50, -0.78, 2.02, 8.8, 5.41],
                               'P': [0.10, 0.00, 0.50, 0.12, 1.99, 10.6, 6.3],
                               'Q': [0.40, 0.09, 0.16, -0.85, 2.17, 9.13, 5.65],
                               'R': [0.50, 0.27, 0.33, -2.53, 2.17, 9.04, 10.76],
                               'S': [0.30, 0.27, 0.50, -0.18, 2.21, 9.15, 5.68],
                               'T': [0.30, 0.45, 0.41, -0.05, 2.09, 9.1, 5.6],
                               'V': [0.50, 1.00, 0.25, 1.08, 2.32, 9.62, 5.96],
                               'W': [0.00, 0.00, 0.00, 0.81, 2.83, 9.39, 5.89],
                               'Y': [0.10, 0.36, 0.08, 0.26, 2.2, 9.11, 5.66]}

        # 将氨基酸列表中的每个氨基酸与其对应索引进行编码，生成字典amino_acid_encoding
        self.amino_acid_encoding = {acid: i for i, acid in enumerate(self.amino_acids)}
        self.label_class = ['H', 'E', 'C']
        # 将标签列表中的每个标签与其对应索引进行编码，生成字典labels_encoding
        self.labels_encoding = {lab: i for i, lab in enumerate(self.label_class)}
        # 序列最大长度
        # self.max_length = max([len(seq.replace(' ', '')) for seq in sequence_list])
        self.max_length = test_max_length

    def __getitem__(self, index):
        # 根据索引获取对应位置的蛋白质序列、标签、smiles序列
        sequence = self.sequences[index]
        label = self.labels[index]
        # 调用类内方法，分别对蛋白质序列进行one-hot编码、将标签嵌入到编码序列、对smiles序列进行编码
        encoded_sequence = self.one_hot_encode_sequence(sequence)

        encoded_properties = self.encode_propertie(sequence)

        # 调用类内方法对标签编码
        encoded_label = self.one_hot_encode_label(label)
        # 读取pssm文件
        pssm_file = f"{index + 1}.csv"
        pssm_path = os.path.join(self.pssm_folder, pssm_file)
        # 调用类内方法，提取PSSM矩阵
        encoded_pssm = self.extract_pssm_data(pssm_path)
        # 左右拼接序列和Pssm
        encoded_fusion1 = torch.cat((encoded_pssm, encoded_sequence), dim=1)
        encoded_fusion = torch.cat((encoded_fusion1, encoded_properties), dim=1)
        # 返回嵌入二级结构后的蛋白质编码、smiles编码-----encoded_sequence, encoded_smiles分别传入进行特征提取
        return encoded_fusion, encoded_label

    def __len__(self):
        # 返回蛋白质序列列表长度==样本数据集数量
        return len(self.sequences)

    def one_hot_encode_sequence(self, sequence):
        encoded_sequence = []
        # 遍历蛋白质序列中的每个氨基酸，并编码成一个维度为20的列表
        for aa in sequence:
            if aa in self.amino_acid_encoding:
                encoded_aa = [0.0] * len(self.amino_acids)
                encoded_aa[self.amino_acid_encoding[aa]] = 1.0
                encoded_sequence.append(encoded_aa)
        # 填充到最大长度
        padding_length = self.max_length - len(encoded_sequence)
        encoded_sequence += [[0.0] * len(self.amino_acids)] * padding_length
        # 返回encoded_sequenc，其中存储编码后的蛋白质序列
        return torch.tensor(encoded_sequence, dtype=torch.float32)

    def encode_propertie(self, sequence):
        encoded_properties = []

        for aa in sequence:
            if aa in self.propertie_data:
                encoded_properties.append(self.propertie_data[aa])
            else:
                encoded_properties.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        padding_length = self.max_length - len(encoded_properties)
        encoded_properties += [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * padding_length

        return torch.tensor(encoded_properties, dtype=torch.float32)

    def extract_pssm_data(self, pssm_file):
        pssm_data = []
        with open(pssm_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过第一行

            # 读取文件中第3列到第22列。提取pssm信息
            for row in reader:
                selected_data = [float(value) for value in row[2:22]]
                pssm_data.append(selected_data)

        padding_length = self.max_length - len(pssm_data)
        pssm_data += [[0.0] * 20] * padding_length

        return torch.tensor(pssm_data, dtype=torch.float32)

    def one_hot_encode_label(self, label):
        encoded_label = []
        # 遍历标签序列中的每个标签，并编码成一个维度为3的列表
        for bb in label:
            if bb in self.labels_encoding:
                encoded_bb = [0] * len(self.label_class)
                encoded_bb[self.labels_encoding[bb]] = 1
                encoded_label.append(encoded_bb)

        # 填充到最大长度
        padding_length = self.max_length - len(encoded_label)
        encoded_label += [[0] * (len(self.label_class))] * padding_length

        # 返回encoded_label，其中存储编码后的标签序列
        return torch.tensor(encoded_label)