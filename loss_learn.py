# coding=utf8
import numpy as np
import scipy.io as sio
from mainfuncSIC_learn import EAOO_latest



def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == '__main__':


    B_ = 30
    T_ = 0.85
    Ps_ = 50
    lr = [0.0001,0.000015, 0.00001, 0.0000095]
    for i in range(len(lr)):
        lr_ = lr[i]
        N = 10
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
        EAOOSIC_time, EAOOSIC_lantency, stop_time_sic,learn_cost_list = EAOO_latest(N, n, E_min, P, E_i, D_i_list, f_i, g_i, B_, T_,Ps_,lr_)


        sio.savemat('./learn/cost_EAOOSIC_' + str(lr[i]) + '.mat',{'learn_cost_list':learn_cost_list})


