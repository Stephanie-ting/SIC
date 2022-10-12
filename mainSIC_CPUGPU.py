# coding=utf8
import numpy as np
import scipy.io as sio
from mainfuncSIC_h import EAOO_latest
from mainfuncSIC_GPU import EAOO_latest_GPU
import tensorflow.compat.v1 as tf
from sic_compute import *

import os
os.environ ['CUDA_DEVISE_ORDER'] = 'PCI_BUS_ID'
os.environ ['CUDA_VISIBLE_DEVICES'] = '0'
# tf.config.experimental.set_visible_devices('0', 'GPU')

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == "__main__":
    EAOOSIC_lantency_list_GPU = []
    EAOOSIC_time_list_GPU = []
    EAOOSIC_lantency_list_CPU = []
    EAOOSIC_time_list_CPU = []
    B_ = 20
    T_ = 1

    for N in range(10, 32, 2):
        n = 3000
        # tips:固定成n个基础值 固定为10J

        E_min = np.mat(abs(np.random.uniform(low=10.0, high=20.0, size=1 * N)).reshape(1, N))
        # 无线设备传输功率
        # tips:固定成n个基础值
        P = np.mat(abs(np.random.uniform(low=0.5, high=0.6, size=1 * N)).reshape(1, N))
        # 计算速率 均匀分布 50-100 [N*1]
        # f_i = np.mat(abs(np.random.uniform(low=80, high=100, size=1 * N)).reshape(1, N))
        f_i = np.mat(abs(np.random.uniform(low=150, high=200, size=1 * N)).reshape(1, N))

        # E_i = np.mat(abs(np.random.normal(loc=23.0, scale=5.0, size=n * N)).reshape(n, N))
        # tips:固定成n个基础值 初始电量[N*1]
        E_i = np.mat(abs(np.random.uniform(low=500.0, high=600.0, size=1 * N)).reshape(1, N))
        # g_i = np.mat(abs(np.random.uniform(low=1.5, high=2.5, size=1 * N)).reshape(1, N))
        # 均匀分布 2-3 np.random.uniform   [N*1]
        g_i = np.mat(abs(np.random.uniform(low=2, high=3, size=1 * N)).reshape(1, N))
        # 任务数据量 均匀分布 50-100 [N*n]
        D_i_list = np.mat(abs(np.random.uniform(low=100, high=150, size=n * N)).reshape(n, N))

        wirelessDevices_, location = create_wireless_device(N, 1, 1, 0.2, 0.35, 0.1)
        server_ = Server((location[0] + location[1]) / 2, (location[2] + location[3]) / 2)

        # EAOO-SIC-GPU算法
        EAOOSIC_time_GPU, EAOOSIC_lantencyG, stop_time_sicG = EAOO_latest_GPU(N, n, E_min, P, E_i, D_i_list, f_i, g_i, wirelessDevices_, server_, B_, T_)
        EAOOSIC_lantency_averageG = EAOOSIC_lantencyG / (stop_time_sicG + 1)  # * 获得具体停止的时间帧stop_time，根据改时间帧得到平均lantency
        EAOOSIC_lantency_list_GPU.append(EAOOSIC_lantency_averageG)
        EAOOSIC_time_list_GPU.append(EAOOSIC_time_GPU)

        # EAOO-SIC-CPU算法
        EAOOSIC_time_CPU, EAOOSIC_lantencyC, stop_time_sicC = EAOO_latest(N, n, E_min, P, E_i, D_i_list, f_i, g_i, wirelessDevices_, server_, B_, T_)
        EAOOSIC_lantency_averageC = EAOOSIC_lantencyC / (stop_time_sicC + 1)  # * 获得具体停止的时间帧stop_time，根据改时间帧到平均lantency
        EAOOSIC_lantency_list_CPU.append(EAOOSIC_lantency_averageC)
        EAOOSIC_time_list_CPU.append(EAOOSIC_time_CPU)



        print('------EAOO-SIC-GPU-----')
        print(EAOOSIC_time_GPU, EAOOSIC_lantency_averageG)
        print('------EAOO-SIC-CPU-----')
        print(EAOOSIC_time_CPU, EAOOSIC_lantency_averageC)
    # print("EAOOSIC_time_GPU of  each 10-30 devices", EAOOSIC_time_list_GPU)
    # print("EAOOsic_lantency_average_GPU of each 10-30 devices,", EAOOSIC_lantency_list_GPU)

    # print("EAOOSIC_time_CPU of  each 10-30 devices", EAOOSIC_time_list_CPU)
    # print("EAOOsic_lantency_average_CPU of each 10-30 devices,", EAOOSIC_lantency_list_CPU)


    save_to_txt(EAOOSIC_time_list_GPU, "./GPU/EAOOSIC_time_GPU1.txt")
    # save_to_txt(EAOOSIC_lantency_list_GPU, "EAOOSIC_lantency_GPU1.txt")
    save_to_txt(EAOOSIC_time_list_CPU, "./GPU/EAOOSIC_time_CPU1.txt")
    #save_to_txt(EAOOSIC_lantency_list_CPU, "EAOOSIC_lantency_CPU1.txt")



