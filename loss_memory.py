import numpy as np
import scipy.io as sio
from mainfuncSIC_memory import EAOO_latest
from sic_compute import *


def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == '__main__':


    B_ = 20
    T_ = 1
    N = 10
    n = 3000
    wirelessDevices_, location = create_wireless_device(N, 1, 1, 0.2, 0.35, 0.1)
    server_ = Server((location[0] + location[1]) / 2, (location[2] + location[3]) / 2)
    EAOOSIC_lantency_list = []
    memory = [64,128, 256, 512, 1024]
    for i in range(len(memory)):
        memory_ = memory[i]


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

        # EAOO-SIC算法
        EAOOSIC_time, EAOOSIC_lantency, stop_time_sic,memory_cost_list = EAOO_latest(N, n, E_min, P, E_i, D_i_list, f_i, g_i, wirelessDevices_, server_, B_, T_,memory_)

        EAOOSIC_lantency_average = EAOOSIC_lantency / (stop_time_sic + 1)  # * 获得具体停止的时间帧stop_time，根据改时间帧到平均lantency
        EAOOSIC_lantency_list.append(EAOOSIC_lantency_average)
        # sio.savemat('./memory/cost_EAOOSIC_' + str(memory[i]) + '.mat',{'memory_cost_list':memory_cost_list})
        sio.savemat('./memory/latency_EAOOSIC_' + str(memory[i]) + '.mat', {'memory_latency_list': EAOOSIC_lantency_average})


