"""
一句话：构建数据集合：
    # 多少萝卜多少坑一个萝卜一个坑
"""
import random
from random import sample
import numpy as np
from torch.utils.data import Dataset


class train_BCIDataset(Dataset):
    def __init__(self, num_data, train_data, win_train, y_label, start_time, down_sample, train_list, channel):
        super(train_BCIDataset, self).__init__()

        x_train, y_train = list(range(num_data)), list(range(num_data))

        # 从90组实验中，随机重复提取num_data组数据
        for i in range(num_data):
            k = sample(train_list, 1)[0]  # 随机选择一组实验
            y_data = y_label[k] - 1  # 获取标签

            time_start = random.randint(35, int(1000 + 35 - win_train))  # 随机时间

            x1 = int(start_time[k] / down_sample) + time_start
            x2 = int(start_time[k] / down_sample) + time_start + win_train
            x_1 = train_data[:, :, x1:x2]
            x_2 = np.reshape(x_1, (4, channel, win_train))

            x_train[i] = x_2.astype(np.float32)  # pytorch 参数是float32，故输入数据的x需要求float32
            y_train[i] = y_data

        self.data = x_train
        self.label = y_train
        self.num_total = num_data

    def __len__(self):
        return self.num_total

    def __getitem__(self, idk):
        return self.data[idk], self.label[idk]


class val_BCIDataset(Dataset):
    def __init__(self, num_data, train_data, win_train, y_label, start_time, down_sample, val_list, channel):
        super(val_BCIDataset, self).__init__()

        x_train, y_train = list(range(num_data)), list(range(num_data))

        # 从10组实验中，随机重复提取256组数据
        for i in range(num_data):
            k = sample(val_list, 1)[0]
            y_data = y_label[k] - 1

            time_start = random.randint(35, int(1000 + 35 - win_train))

            x1 = int(start_time[k] / down_sample) + time_start
            x2 = int(start_time[k] / down_sample) + time_start + win_train
            x_1 = train_data[:, :, x1:x2]
            x_2 = np.reshape(x_1, (4, channel, win_train))

            x_train[i] = x_2.astype(np.float32)
            y_train[i] = y_data

        self.data = x_train
        self.label = y_train
        self.num_total = num_data

    def __len__(self):
        return self.num_total

    def __getitem__(self, idk):
        return self.data[idk], self.label[idk]


class test_BCIDataset(Dataset):
    def __init__(self, num_data, test_data, win_train, y_label, start_time, down_sample, channel):
        super(test_BCIDataset, self).__init__()
        x_test, y_test = list(range(num_data)), list(range(num_data))

        # 从100组实验中，随机重复提取1000组数据
        for i in range(num_data):
            k = random.randint(0, (y_label.shape[0] - 1))  # 随机选择一组实验
            y_data = y_label[k] - 1  # 获取标签

            time_start = random.randint(35, int(1000 + 35 - win_train))  # 随机时间

            x1 = int(start_time[k] / down_sample) + time_start
            x2 = int(start_time[k] / down_sample) + time_start + win_train
            x_1 = test_data[:, :, x1:x2]
            x_2 = np.reshape(x_1, (4, channel, win_train))

            x_test[i] = x_2.astype(np.float32)  # pytorch 参数是float32，故输入数据的x需要求float32
            y_test[i] = y_data

        self.data = x_test
        self.label = y_test
        self.num_total = num_data

    def __len__(self):
        return self.num_total

    def __getitem__(self, idk):
        return self.data[idk], self.label[idk]

