# coding=utf8
import numpy as np
from mainfuncEAOO import EAOO_latest_serial
from mainfuncSIC import EAOO_latest
from mainfuncDROO import DROO_latest_serial
from mainfunlocal import EAOO_local
import matplotlib.pyplot as plt
import scipy.io as sio

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == "__main__":
    EAOOSIC_lantency_list = []
    EAOOSIC_time_list = []
    EAOO_lantency_list = []
    EAOO_time_list = []
    local_lantency_list = []
    local_time_list = []
    DROO_lantency_list = []
    DROO_time_list = []
    DROO_disrunnable_list = []

    B_ = 30
    T_ = 0.85

    for N in range(10, 32, 2):
        n = 3000

        E_min = np.mat(abs(np.random.uniform(low=10.0, high=20.0, size=1 * N)).reshape(1, N))
        # 无线设备传输功率
        # tips:固定成n个基础值
        P = np.mat(abs(np.random.uniform(low=0.5, high=0.6, size=1 * N)).reshape(1, N))
        # 计算速率 均匀分布 50-100 [N*1]
        # f_i = np.mat(abs(np.random.uniform(low=80, high=100, size=1 * N)).reshape(1, N))
        f_i = np.mat(abs(np.random.uniform(low=150, high=200, size=1 * N)).reshape(1, N))

        # E_i = np.mat(abs(np.random.normal(loc=23.0, scale=5.0, size=n * N)).reshape(n, N))
        # tips:固定成n个基础值 初始电量[N*1]
        E_i = np.mat(abs(np.random.uniform(low=500.0, high=600.0, size=n * N)).reshape(n, N))
        # g_i = np.mat(abs(np.random.uniform(low=1.5, high=2.5, size=1 * N)).reshape(1, N))
        # 均匀分布 2-3 np.random.uniform   [N*1]
        g_i = np.mat(abs(np.random.uniform(low=2, high=3, size=n * N)).reshape(n, N))
        # 任务数据量 均匀分布 50-100 [N*n]
        D_i_list = np.mat(abs(np.random.uniform(low=50, high=150, size=n * N)).reshape(n, N))

        # EAOO-SIC算法
        EAOOSIC_time, EAOOSIC_lantency, stop_time_sic = EAOO_latest(N, n, E_min, P, E_i, D_i_list, f_i, g_i, B_, T_)
        EAOOSIC_lantency_average = EAOOSIC_lantency / (stop_time_sic + 1)  # * 获得具体停止的时间帧stop_time，根据改时间帧到平均lantency
        EAOOSIC_lantency_list.append(EAOOSIC_lantency_average)
        EAOOSIC_time_list.append(EAOOSIC_time)
        
        #EAOO-串行  
        EAOO_time, EAOO_lantency, stop_time =  EAOO_latest_serial(N, n, E_min, P, E_i, D_i_list, f_i, g_i, B_, T_)
        EAOO_lantency_average = EAOO_lantency / (stop_time + 1)
        EAOO_lantency_list.append(EAOO_lantency_average)
        EAOO_time_list.append(EAOO_time)

        # 完全本地
        local_time, local_lantency, stop_time = EAOO_local(N, n, E_min, E_i, D_i_list, f_i, g_i, T_)
        local_lantency_average = local_lantency / (stop_time + 1)
        local_lantency_list.append(local_lantency_average)
        local_time_list.append(local_time)

        # DROO算法
        DROO_time, DROO_lantency, stop_time, disrunnable_times = DROO_latest_serial(N, n, E_min, P, E_i, D_i_list, f_i, g_i, B_, T_)
        DROO_lantency_average = DROO_lantency / (stop_time + 1 - disrunnable_times)  # * 获得具体停止的时间帧stop_time，根据改时间帧得到平均lantency
        DROO_lantency_list.append(DROO_lantency_average)
        DROO_time_list.append(DROO_time)
        DROO_disrunnable_list.append(disrunnable_times)

    print('------EAOO-SIC-----')
    print("EAOOSIC_time of  each 10-30 devices", EAOOSIC_time_list)
    print("EAOOsic_lantency_average of each 10-30 devices,", EAOOSIC_lantency_list)

    print('------EAOO---------')
    print("EAOO_time of  each 10-30 devices", EAOO_time_list)
    print("EAOO_lantency_average of each 10-30 devices,", EAOO_lantency_list)

    print('------DROO---------')
    print("DROO_time of  each 10-30 devices", DROO_time_list)
    print("DROO_lantency_average of each 10-30 devices,", DROO_lantency_list)

    print('------local--------')
    print("local_time of  each 10-30 devices", local_time_list)
    print("local_lantency_average of each 10-30 devices,", local_lantency_list)

    save_to_txt(EAOOSIC_time_list, "EAOOSIC_time.txt")
    save_to_txt(EAOOSIC_lantency_list, "EAOOSIC_lantency.txt")
    save_to_txt(EAOO_time_list, "EAOO_time.txt")
    save_to_txt(EAOO_lantency_list, "EAOO_lantency.txt")
    save_to_txt(local_time_list, "local_time.txt")
    save_to_txt(local_lantency_list, "local_lantency.txt")
    save_to_txt(DROO_time_list, "DROO_time.txt")
    save_to_txt(DROO_lantency_list, "DROO_lantency.txt")
    save_to_txt(DROO_disrunnable_list,"DROO_disrunnable_list")



