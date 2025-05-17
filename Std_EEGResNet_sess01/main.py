"""
EEEResNet网络训练主逻辑
Args:
    1 时间窗口选择为xxx;
    2 网络性能评价指标：accuracy, precision, recall and F1-score;
------------------------------------------------------------------------------
Std标准文件-网络训练的标准文件
    1 网络训练有3个阶段：@训练阶段-tra，@验证阶段-val，@测试阶段-tes;
    2 网络训练有三个循环：@受试者循环-sub，@世代循环-epo，@批次循环-bth, @全局global-glo;
    3 需要保存的文件用_log结尾; 中继变量以mid_开头;
    4 Rules:
        #1 变量命名示例1: data_tra_bth;
        #2 变量命名示例1: loss_tra_sub_log;
    5 常见缩写:
        #1 time->tim; #2 window->win; #3 number->num;
        #4 idx->index; #5 predict->pdt; #6 target->tgt;
        #7 middle->mid; #8 iteration->iter;
"""
import os
import csv
import codecs
import torch
import numpy as np
from EEGResNet import ResidualBlock, ResNet
import scipy.io as scio
from scipy import signal
from random import sample
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import train_BCIDataset, val_BCIDataset, test_BCIDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# @获取滤波后的训练数据，标签和起始时间
def get_train_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24, path, down_sample):
    data = scio.loadmat(path)  # 读取原始数据

    # 下采样与通道选择
    x_data = data['EEG_SSVEP_train']['x'][0][0][::down_sample]
    c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
    train_data = x_data[:, c]
    train_label = data['EEG_SSVEP_train']['y_dec'][0][0][0]
    train_start_time = data['EEG_SSVEP_train']['t'][0][0][0]

    # @ 滤波1
    channel_data_list1 = []
    for i in range(train_data.shape[1]):
        b1, a1 = signal.butter(6, [wn11, wn21], 'bandpass')
        filtedData1 = signal.filtfilt(b1, a1, train_data[:, i])
        channel_data_list1.append(filtedData1)
    channel_data_list1 = np.array(channel_data_list1)

    # @ 滤波2
    channel_data_list2 = []
    for i in range(train_data.shape[1]):
        b2, a2 = signal.butter(6, [wn12, wn22], 'bandpass')
        filtedData2 = signal.filtfilt(b2, a2, train_data[:, i])
        channel_data_list2.append(filtedData2)
    channel_data_list2 = np.array(channel_data_list2)

    # @ 滤波3
    channel_data_list3 = []
    for i in range(train_data.shape[1]):
        b3, a3 = signal.butter(6, [wn13, wn23], 'bandpass')
        filtedData3 = signal.filtfilt(b3, a3, train_data[:, i])
        channel_data_list3.append(filtedData3)
    channel_data_list3 = np.array(channel_data_list3)

    # @ 滤波4
    channel_data_list4 = []
    for i in range(train_data.shape[1]):
        b4, a4 = signal.butter(6, [wn14, wn24], 'bandpass')
        filtedData4 = signal.filtfilt(b4, a4, train_data[:, i])
        channel_data_list4.append(filtedData4)
    channel_data_list4 = np.array(channel_data_list4)

    return channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4, train_label, train_start_time


# @获取滤波后的测试数据、标签和起始时间
def get_test_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24, path, down_sample):
    data = scio.loadmat(path)

    # 下采样与通道选择
    x_data = data['EEG_SSVEP_test']['x'][0][0][::down_sample]
    c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
    test_data = x_data[:, c]
    test_label = data['EEG_SSVEP_test']['y_dec'][0][0][0]
    test_start_time = data['EEG_SSVEP_test']['t'][0][0][0]

    # @ 滤波1
    channel_data_list1 = []
    for i in range(test_data.shape[1]):
        b, a = signal.butter(6, [wn11, wn21], 'bandpass')
        filtedData = signal.filtfilt(b, a, test_data[:, i])
        channel_data_list1.append(filtedData)
    channel_data_list1 = np.array(channel_data_list1)

    # @ 滤波2
    channel_data_list2 = []
    for i in range(test_data.shape[1]):
        b2, a2 = signal.butter(6, [wn12, wn22], 'bandpass')
        filtedData2 = signal.filtfilt(b2, a2, test_data[:, i])
        channel_data_list2.append(filtedData2)
    channel_data_list2 = np.array(channel_data_list2)

    # @ 滤波3
    channel_data_list3 = []
    for i in range(test_data.shape[1]):
        b3, a3 = signal.butter(6, [wn13, wn23], 'bandpass')
        filtedData3 = signal.filtfilt(b3, a3, test_data[:, i])
        channel_data_list3.append(filtedData3)
    channel_data_list3 = np.array(channel_data_list3)

    # @ 滤波4
    channel_data_list4 = []
    for i in range(test_data.shape[1]):
        b4, a4 = signal.butter(6, [wn14, wn24], 'bandpass')
        filtedData4 = signal.filtfilt(b4, a4, test_data[:, i])
        channel_data_list4.append(filtedData4)
    channel_data_list4 = np.array(channel_data_list4)

    return channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4, test_label, test_start_time


# @保存csv格式数据-注意此部分有待完善！
def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")


if __name__ == '__main__':
    # @GPU加速
    # print(torch.cuda.device_count())  # 打印当前设备GPU数量，此笔记本只有1个GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device is', device)

    # @1 参数与滤波器设置
    # --------------------------------------------#
    # 训练世代、数据量、batchsize及学习率设置
    train_epoch = 2000  # 训练的世代
    num_data = 2000
    batchsize = 250
    lr = 8e-4

    # 下采样与滤波设置
    down_sample = 4  # 下采样
    fs = 1000 / down_sample
    channel = 9  # 通道

    f_down1 = 3  # 第一个滤波器
    f_up1 = 14
    wn11 = 2 * f_down1 / fs
    wn21 = 2 * f_up1 / fs

    f_down2 = 9  # 第二个滤波器
    f_up2 = 26
    wn12 = 2 * f_down2 / fs
    wn22 = 2 * f_up2 / fs

    f_down3 = 14  # 第三个滤波器
    f_up3 = 38
    wn13 = 2 * f_down3 / fs
    wn23 = 2 * f_up3 / fs

    f_down4 = 19  # 第四个滤波器
    f_up4 = 50
    wn14 = 2 * f_down4 / fs
    wn24 = 2 * f_up4 / fs
    # --------------------------------------------#

    # Subject from 1 to 9
    for subject_idx in range(1, 10):
        # @2 数据集设置
        # -------------------------------------------------------------------------------------------------------------#
        # 受试者选择
        path = './sess01/sess01_subj0%d_EEG_SSVEP.mat' % subject_idx

        # 获取滤波数据、标签和起始时间
        data1, data2, data3, data4, label, start_time = get_train_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24,
                                                                       path, down_sample)
        data = [data1, data2, data3, data4]  # 数据的聚合
        data = np.array(data)

        # 时间窗口选择
        win_train = 512  # 时间窗口对应的帧数
        train_list = list(range(100))  # 对100次实验，随机划分
        val_list = sample(train_list, 10)
        train_list = [train_list[i] for i in range(len(train_list)) if (i not in val_list)]

        # 获取测试滤波数据、标签和起始时间
        data1_test, data2_test, data3_test, data4_test, label_test, start_time_test = get_test_data(wn11, wn21, wn12,
                                                                                                    wn22, wn13, wn23,
                                                                                                    wn14, wn24, path,
                                                                                                    down_sample)
        data_test = [data1_test, data2_test, data3_test, data4_test]
        data_test = np.array(data_test)

        # 数据产生器
        train_dataset = train_BCIDataset(num_data, data, win_train, label, start_time, down_sample, train_list, channel)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batchsize, num_workers=1, pin_memory=True,
                         drop_last=True)

        val_dataset = val_BCIDataset(num_data, data, win_train, label, start_time, down_sample, val_list, channel)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batchsize, num_workers=1, pin_memory=True,
                             drop_last=True)

        test_dataset = test_BCIDataset(num_data, data_test, win_train, label_test, start_time_test, down_sample,
                                       channel)
        gen_test = DataLoader(test_dataset, shuffle=True, batch_size=batchsize, num_workers=1, pin_memory=True,
                              drop_last=True)
        # -------------------------------------------------------------------------------------------------------------#

        # @3 网络设置
        # -------------------------------------------------------------------------------------------------------------#
        # @ 网络损失优化器初始化
        net = ResNet(ResidualBlock)
        net.to(device)
        loss_f = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr, weight_decay=0.01)  # 对参数进行正则化weight_decay, 防止过拟合
        # optimizer = optim.Adam(net.parameters(), lr)
        # -------------------------------------------------------------------------------------------------------------#

        # @4 循环训练
        # -------------------------------------------------------------------------------------------------------------#
        history_train_loss = [[]]  # 保存的损失历史数据
        history_val_loss = [[]]
        history_test_loss = [[]]
        history_av_acc = [[]]  # 保存的准确性历史数据
        history_av_acc_test = [[]]

        for epoch in range(train_epoch):
            sum_loss_epoch_train = []  # 单世代损失缓存
            sum_loss_epoch_val = []
            sum_loss_epoch_test = []
            acc_list_val = []  # 单世代准确性缓存
            acc_list_test = []

            epoch_size = num_data / batchsize  # 单世代循环次数

            # 训练模式
            net.train()
            # print('Start train')
            for iteration, batch in enumerate(gen):  # iteration是批次，batch是每批次的数据
                if iteration >= epoch_size:
                    break
                data_train, targets = batch[0], batch[1]
                data_train, targets = data_train.to(device), targets.to(device)

                optimizer.zero_grad()  # 清零梯度
                outputs = net(data_train)  # 前向传播
                loss_t = loss_f(outputs, targets.long())  # 损失计算
                sum_loss_epoch_train.append(loss_t)
                loss_t.backward()
                optimizer.step()

            train_loss = sum(sum_loss_epoch_train) / epoch_size
            history_train_loss.append([epoch, train_loss.data.cpu().numpy()])  # 先转成普通tensor，再转成numpy形式
            # print('End train')

            # 验证模式
            net.eval()
            # print('Start validation')
            for iteration, batch in enumerate(gen_val):  # iteration是批次，batch是每批次的数据
                if iteration >= epoch_size:
                    break
                data_train_val, targets_val = batch[0], batch[1]
                data_train_val, targets_val = data_train_val.to(device), targets_val.to(device)

                with torch.no_grad():
                    optimizer.zero_grad()  # 清零梯度
                    outputs = net(data_train_val)  # 前向传播
                    loss_v = loss_f(outputs, targets_val.long())  # 损失计算
                    sum_loss_epoch_val.append(loss_v)  # 损失保存

                    # 计算准确性
                    y_true = targets_val.cpu().numpy()
                    a, b = 0, 0
                    for i in range(batchsize - 1):
                        y_pred_label = np.argmax(outputs.cpu().numpy()[i])
                        if y_true[i] == y_pred_label:
                            a += 1
                        else:
                            b += 1
                    acc = a / (a + b)
                    acc_list_val.append(acc)
            av_acc_epoch = np.mean(acc_list_val)
            val_loss = sum(sum_loss_epoch_val) / epoch_size
            history_val_loss.append([epoch, val_loss.data.cpu().numpy()])  # 先转成普通tensor，再转成numpy形式
            history_av_acc.append([epoch, av_acc_epoch])
            # print('Finish validation')

            # 测试模式
            net.eval()
            # print('Start test')
            for iteration, batch in enumerate(gen_test):  # iteration是批次，batch是每批次的数据
                if iteration >= epoch_size:
                    break
                data_train_test, targets_test = batch[0], batch[1]
                data_train_test, targets_test = data_train_test.to(device), targets_test.to(device)

                with torch.no_grad():
                    optimizer.zero_grad()  # 清零梯度
                    outputs = net(data_train_test)  # 前向传播
                    loss_tt = loss_f(outputs, targets_test.long())  # 损失计算
                    sum_loss_epoch_test.append(loss_tt)

                    # 计算准确性
                    y_true = targets_test.cpu().numpy()
                    a, b = 0, 0
                    for i in range(batchsize - 1):
                        y_pred_label = np.argmax(outputs.cpu().numpy()[i])
                        if y_true[i] == y_pred_label:
                            a += 1
                        else:
                            b += 1
                    acc = a / (a + b)
                    acc_list_test.append(acc)
            av_acc_epoch_test = np.mean(acc_list_test)
            test_loss = sum(sum_loss_epoch_test) / epoch_size
            history_test_loss.append([epoch, test_loss.data.cpu().numpy()])  # 先转成普通tensor，再转成numpy形式
            history_av_acc_test.append([epoch, av_acc_epoch_test])
            # print('Finish validation')

            print('subject is %d, epoch is %d, train_loss is %d, val_loss is %d, test_loss is %d, val_acc is %d, '
                  'test_acc is %d', (subject_idx, epoch, train_loss.cpu().detach().numpy(), val_loss, test_loss, av_acc_epoch,
                                     av_acc_epoch_test))  # 信息反馈

        path_history_val_loss = './logs/Sub0%d_history_val_loss.csv' % subject_idx
        path_history_train_loss = './logs/Sub0%d_history_train_loss.csv' % subject_idx
        path_history_test_loss = './logs/Sub0%d_history_test_loss.csv' % subject_idx
        path_history_av_acc = './logs/Sub0%d_history_av_acc.csv' % subject_idx
        path_history_av_acc_test = './logs/Sub0%d_history_av_acc_test.csv' % subject_idx
        data_write_csv(path_history_val_loss, history_val_loss)  # 损失历史数据
        data_write_csv(path_history_train_loss, history_train_loss)
        data_write_csv(path_history_test_loss, history_test_loss)
        data_write_csv(path_history_av_acc, history_av_acc)  # 准确性历史数据
        data_write_csv(path_history_av_acc_test, history_av_acc_test)

    # Subject from 10 to 54
    for subject_idx in range(10, 55):
        # @2 数据集设置
        # -------------------------------------------------------------------------------------------------------------#
        # 受试者选择
        path = './sess01/sess01_subj%d_EEG_SSVEP.mat' % subject_idx

        # 获取滤波数据、标签和起始时间
        data1, data2, data3, data4, label, start_time = get_train_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24,
                                                                       path, down_sample)
        data = [data1, data2, data3, data4]  # 数据的聚合
        data = np.array(data)

        # 时间窗口选择
        win_train = 512  # 时间窗口对应的帧数
        train_list = list(range(100))  # 对100次实验，随机划分
        val_list = sample(train_list, 10)
        train_list = [train_list[i] for i in range(len(train_list)) if (i not in val_list)]

        # 获取测试滤波数据、标签和起始时间
        data1_test, data2_test, data3_test, data4_test, label_test, start_time_test = get_test_data(wn11, wn21, wn12,
                                                                                                    wn22, wn13, wn23,
                                                                                                    wn14, wn24, path,
                                                                                                    down_sample)
        data_test = [data1_test, data2_test, data3_test, data4_test]
        data_test = np.array(data_test)

        # 数据产生器
        train_dataset = train_BCIDataset(num_data, data, win_train, label, start_time, down_sample, train_list, channel)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batchsize, num_workers=1, pin_memory=True,
                         drop_last=True)

        val_dataset = val_BCIDataset(num_data, data, win_train, label, start_time, down_sample, val_list, channel)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batchsize, num_workers=1, pin_memory=True,
                             drop_last=True)

        test_dataset = test_BCIDataset(num_data, data_test, win_train, label_test, start_time_test, down_sample,
                                       channel)
        gen_test = DataLoader(test_dataset, shuffle=True, batch_size=batchsize, num_workers=1, pin_memory=True,
                              drop_last=True)
        # -------------------------------------------------------------------------------------------------------------#

        # @3 网络设置
        # -------------------------------------------------------------------------------------------------------------#
        # @ 网络损失优化器初始化
        net = ResNet(ResidualBlock)
        net.to(device)
        loss_f = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr, weight_decay=0.01)  # 对参数进行正则化weight_decay, 防止过拟合
        # optimizer = optim.Adam(net.parameters(), lr)
        # -------------------------------------------------------------------------------------------------------------#

        # @4 循环训练
        # -------------------------------------------------------------------------------------------------------------#
        history_train_loss = [[]]  # 保存的损失历史数据
        history_val_loss = [[]]
        history_test_loss = [[]]
        history_av_acc = [[]]  # 保存的准确性历史数据
        history_av_acc_test = [[]]

        for epoch in range(train_epoch):
            sum_loss_epoch_train = []  # 单世代损失缓存
            sum_loss_epoch_val = []
            sum_loss_epoch_test = []
            acc_list_val = []  # 单世代准确性缓存
            acc_list_test = []

            epoch_size = num_data / batchsize  # 单世代循环次数

            # 训练模式
            net.train()
            # print('Start train')
            for iteration, batch in enumerate(gen):  # iteration是批次，batch是每批次的数据
                if iteration >= epoch_size:
                    break
                data_train, targets = batch[0], batch[1]
                data_train, targets = data_train.to(device), targets.to(device)

                optimizer.zero_grad()  # 清零梯度
                outputs = net(data_train)  # 前向传播
                loss_t = loss_f(outputs, targets.long())  # 损失计算
                sum_loss_epoch_train.append(loss_t)
                loss_t.backward()
                optimizer.step()

            train_loss = sum(sum_loss_epoch_train) / epoch_size
            history_train_loss.append([epoch, train_loss.data.cpu().numpy()])  # 先转成普通tensor，再转成numpy形式
            # print('End train')

            # 验证模式
            net.eval()
            # print('Start validation')
            for iteration, batch in enumerate(gen_val):  # iteration是批次，batch是每批次的数据
                if iteration >= epoch_size:
                    break
                data_train_val, targets_val = batch[0], batch[1]
                data_train_val, targets_val = data_train_val.to(device), targets_val.to(device)

                with torch.no_grad():
                    optimizer.zero_grad()  # 清零梯度
                    outputs = net(data_train_val)  # 前向传播
                    loss_v = loss_f(outputs, targets_val.long())  # 损失计算
                    sum_loss_epoch_val.append(loss_v)  # 损失保存

                    # 计算准确性
                    y_true = targets_val.cpu().numpy()
                    a, b = 0, 0
                    for i in range(batchsize - 1):
                        y_pred_label = np.argmax(outputs.cpu().numpy()[i])
                        if y_true[i] == y_pred_label:
                            a += 1
                        else:
                            b += 1
                    acc = a / (a + b)
                    acc_list_val.append(acc)
            av_acc_epoch = np.mean(acc_list_val)
            val_loss = sum(sum_loss_epoch_val) / epoch_size
            history_val_loss.append([epoch, val_loss.data.cpu().numpy()])  # 先转成普通tensor，再转成numpy形式
            history_av_acc.append([epoch, av_acc_epoch])
            # print('Finish validation')

            # 测试模式
            net.eval()
            # print('Start test')
            for iteration, batch in enumerate(gen_test):  # iteration是批次，batch是每批次的数据
                if iteration >= epoch_size:
                    break
                data_train_test, targets_test = batch[0], batch[1]
                data_train_test, targets_test = data_train_test.to(device), targets_test.to(device)

                with torch.no_grad():
                    optimizer.zero_grad()  # 清零梯度
                    outputs = net(data_train_test)  # 前向传播
                    loss_tt = loss_f(outputs, targets_test.long())  # 损失计算
                    sum_loss_epoch_test.append(loss_tt)

                    # 计算准确性
                    y_true = targets_test.cpu().numpy()
                    a, b = 0, 0
                    for i in range(batchsize - 1):
                        y_pred_label = np.argmax(outputs.cpu().numpy()[i])
                        if y_true[i] == y_pred_label:
                            a += 1
                        else:
                            b += 1
                    acc = a / (a + b)
                    acc_list_test.append(acc)
            av_acc_epoch_test = np.mean(acc_list_test)
            test_loss = sum(sum_loss_epoch_test) / epoch_size
            history_test_loss.append([epoch, test_loss.data.cpu().numpy()])  # 先转成普通tensor，再转成numpy形式
            history_av_acc_test.append([epoch, av_acc_epoch_test])
            # print('Finish validation')

            print('subject is %d, epoch is %d, train_loss is %d, val_loss is %d, test_loss is %d, val_acc is %d, '
                  'test_acc is %d', (subject_idx, epoch, train_loss, val_loss, test_loss, av_acc_epoch,
                                     av_acc_epoch_test))  # 信息反馈

        path_history_val_loss = './logs/Sub0%d_history_val_loss.csv' % subject_idx
        path_history_train_loss = './logs/Sub0%d_history_train_loss.csv' % subject_idx
        path_history_test_loss = './logs/Sub0%d_history_test_loss.csv' % subject_idx
        path_history_av_acc = './logs/Sub0%d_history_av_acc.csv' % subject_idx
        path_history_av_acc_test = './logs/Sub0%d_history_av_acc_test.csv' % subject_idx
        data_write_csv(path_history_val_loss, history_val_loss)  # 损失历史数据
        data_write_csv(path_history_train_loss, history_train_loss)
        data_write_csv(path_history_test_loss, history_test_loss)
        data_write_csv(path_history_av_acc, history_av_acc)  # 准确性历史数据
        data_write_csv(path_history_av_acc_test, history_av_acc_test)
