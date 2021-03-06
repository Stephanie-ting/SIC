import numpy as np

from mainfuncEAOO import EAOO_latest_serial
from mainfuncSIC import EAOO_latest
from mainfuncDROO import DROO_latest_serial
from mainfunlocal import EAOO_local

import scipy.io as sio

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == '__main__':
    EAOOSIC_B_latency_list = []
    EAOO_B_latency_list = []
    DROO_B_latency_list = []
    local_B_latency_list = []
    # B_ = 30
    T_ = 1

    for B in range(10, 32, 2):
        N = 10
        n = 5000

        E_min = np.mat(abs(np.random.uniform(low=10.0, high=20.0, size=1 * N)).reshape(1, N))
        # 无线设备传输功率
        P = np.mat(abs(np.random.uniform(low=0.5, high=0.6, size=1 * N)).reshape(1, N))
        # 计算速率
        f_i = np.mat(abs(np.random.uniform(low=150, high=200, size=1 * N)).reshape(1, N))
        E_i = np.mat(abs(np.random.uniform(low=500.0, high=600.0, size=1 * N)).reshape(1, N))
        g_i = np.mat(abs(np.random.uniform(low=2, high=3, size=1 * N)).reshape(1, N))
        # 任务数据量 均匀分布 50-100 [N*n]
        D_i_list = np.mat(abs(np.random.uniform(low=50, high=100, size=n * N)).reshape(n, N))

        # EAOO-SIC算法
        EAOOSIC_time, EAOOSIC_lantency, stop_time_sic = EAOO_latest(N, n, E_min, P, E_i, D_i_list, f_i, g_i, B, T_)
        EAOOSIC_lantency_average = EAOOSIC_lantency / (stop_time_sic + 1)  # * 获得具体停止的时间帧stop_time，根据改时间帧得到平均lantency
        EAOOSIC_B_latency_list.append(EAOOSIC_lantency_average)
        # EAOO-串行
        EAOO_time, EAOO_lantency, stop_time_ori = EAOO_latest_serial(N, n, E_min, P, E_i, D_i_list, f_i, g_i, B, T_)
        EAOO_lantency_average = EAOO_lantency / (stop_time_ori + 1)
        EAOO_B_latency_list.append(EAOO_lantency_average)
        # DROO算法
        DROO_time, DROO_lantency, stop_time_droo, disrunnable_times = DROO_latest_serial(N, n, E_min, P, E_i, D_i_list, f_i, g_i, B, T_)
        DROO_lantency_average = DROO_lantency / (stop_time_droo + 1 - disrunnable_times)  # * 获得具体停止的时间帧stop_time，根据改时间帧得到平均lantency
        DROO_B_latency_list.append(DROO_lantency_average)
        # 完全本地
        local_time, local_lantency, stop_time_local = EAOO_local(N, n, E_min, E_i, D_i_list, f_i, g_i, B,T_)
        local_lantency_average = local_lantency / (stop_time_local + 1)
        local_B_latency_list.append(local_lantency_average)

    save_to_txt(EAOOSIC_B_latency_list, 'EAOOSIC_B_latency_list.txt')
    save_to_txt(EAOO_B_latency_list, 'EAOO_B_latency_list.txt')
    save_to_txt(DROO_B_latency_list, 'DROO_B_latency_list.txt')
    save_to_txt(local_B_latency_list, 'local_B_latency_list.txt')
