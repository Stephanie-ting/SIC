# coding=utf8
#  #################################################################
#  Deep Reinforcement Learning for Online Ofﬂoading in Wireless Powered Mobile-Edge Computing Networks
#
#  This file contains the main code of DROO. It loads the training samples saved in ./data/data_#.mat, splits the samples into two parts (training and testing data constitutes 80% and 20%), trains the DNN with training and validation samples, and finally tests the DNN with test data.
#
#  Input: ./data/data_#.mat
#    Data samples are generated according to the CD method presented in [2]. There are 30,000 samples saved in each ./data/data_#.mat, where # is the user number. Each data sample includes
#  -----------------------------------------------------------------
#  |       wireless channel gain           |    input_h            |
#  -----------------------------------------------------------------
#  |       computing mode selection        |    output_mode        |
#  -----------------------------------------------------------------
#  |       energy broadcasting parameter   |    output_a           |
#  -----------------------------------------------------------------
#  |     transmit time of wireless device  |    output_tau         |
#  -----------------------------------------------------------------
#  |      weighted sum computation rate    |    output_obj         |
#  -----------------------------------------------------------------
#
#
#  References:
#  [1] 1. Liang Huang, Suzhi Bi, and Ying-Jun Angela Zhang, "Deep Reinforcement Learning for Online Offloading in Wireless Powered Mobile-Edge Computing Networks," in IEEE Transactions on Mobile Computing, early access, 2019, DOI:10.1109/TMC.2019.2928811.
#  [2] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” IEEE Trans. Wireless Commun., vol. 17, no. 6, pp. 4177-4190, Jun. 2018.
#
# version 1.0 -- July 2018. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################


import scipy.io as sio  # import scipy.io for .mat file I/
import numpy as np  # import numpy
import time


def EAOO_local(N_, n_, E_min_, E_i_, D_i_list_, f_i_, g_i_, B, T_=2, Ps_=50):

    def localEnergyCost(a, f_i, D_i, g):
        return 1.0 * a * f_i * f_i * D_i * g

    N = N_  # number of users
    n = n_  # number of time frames

    T = T_

    amin = 1e-27
    flagWD = [0 for index in range(N)]
    # print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,n,K,decoder_mode, Memory, Delta))
    # Load data
    if N in [5, 6, 7, 8, 9, 10, 20, 30]:
        channel0 = sio.loadmat('./data/data_%d' % N)['input_h']
        # rate = sio.loadmat('./data/data_%d' % N)['output_obj']  # this rate is only used to plot figures; never used to train DROO.
    else:
        channel_temp = sio.loadmat('./data/data_%d' % 30)['input_h']
        channel0 = channel_temp[:, 0:N]
    # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
    channel = channel0 * 1000000

    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n))  # training data size

    start_time = time.time()

    print('The algorithm Local with', N, 'WDs begin.')
    # 初始能量
    E_i = E_i_[0, :].tolist()
    # 计算速率
    f_i = f_i_[0, :].tolist()[0]
    print("f_i，",f_i)
    # CPU周期
    g_i = g_i_[0, :].tolist()
    E_min = E_min_[0, :].tolist()
    WD_m = [0 for num in range(N)]
    totallantency = 0
    # print('The function all WDs execute local with', N, 'WDs begin.')

    E_flag = False  # * 如果能力不足了就break
    stop_time = n - 1  # * 算法在哪个时间帧停止
    for i in range(n):



        if i % (n // 10) == 0:
            pass

        if i < n - num_test:
            # training
            i_idx = i % split_idx
        else:
            # test
            i_idx = i - n + num_test + split_idx

        h = channel[i_idx, :]
        h0 = channel0[i_idx, :]
        D_i_list = D_i_list_[i, :].tolist()
        for fl in range(N):  # 上轮该无线设备未执行完任务，本轮不予分配新的任务
            if flagWD[fl] > T:
                D_i_list[fl] = 0
                flagWD[fl] -= T  # 本轮执行完后所需的时间帧数减

        # 各无线设备能量更新
        for index in range(N):

            C_local = localEnergyCost(amin, f_i[index], D_i_list[index], g_i[index])

            if E_i[index] < E_min[index]:
                stop_time = i
                stop_time = min(stop_time, i)  # 算法在第 i 个时间帧停止
                E_flag = True
                break

            E_i[index]  -= C_local
            if D_i_list[index] != 0:
                flagWD[index] = D_i_list[index] * g_i[index] / f_i[index]
            totallantency += D_i_list[index] * g_i[index] / f_i[index]

        if E_flag:
            break


    total_time = time.time() - start_time

    print(N, '个 WDs',"算法停止的时间帧(0-2999):", stop_time)
    return total_time, totallantency,stop_time


if __name__ == "__main__":

    local_lantency_list = []
    local_time_list = []

    B_ = 30
    T_ = 2
    Ps_ = 50
    for N in range(10, 32, 2):
        n = 3000
        # tips:固定成n个基础值 固定为10J
        E_min = np.mat(abs(np.random.uniform(low=10.0, high=20.0, size=1 * N)).reshape(1, N))
        # 无线设备传输功率
        # tips:固定成n个基础值
        P = np.mat(abs(np.random.uniform(low=0.5, high=0.6, size=1 * N)).reshape(1, N))
        # E_i = np.mat(abs(np.random.normal(loc=23.0, scale=5.0, size=n * N)).reshape(n, N))
        # tips:固定成n个基础值 初始电量[N*1]
        E_i = np.mat(abs(np.random.uniform(low=500.0, high=600.0, size=1 * N)).reshape(1, N))
        # 无线设备能量采集系数 初始系数[N*1] 随机分布0.5-1 [N*1]
        # eh_i = np.mat(abs(np.random.uniform(low=0.6, high=0.8, size=1 * N)).reshape(1, N))
        # D_i_list = np.mat(abs(np.random.normal(loc=80.0, scale=45.0, size=n * N)).reshape(n, N))
        # 任务数据量 均匀分布 50-100 [N*n]
        D_i_list = np.mat(abs(np.random.uniform(low=100, high=150, size=n * N)).reshape(n, N))
        # 计算速率 均匀分布 50-100 [N*1]
        # f_i = np.mat(abs(np.random.uniform(low=80, high=100, size=1 * N)).reshape(1, N))
        f_i = np.mat(abs(np.random.uniform(low=50, high=150, size=1 * N)).reshape(1, N))
        # CPU周期 [N*1]
        # g_i = np.mat(abs(np.random.uniform(low=1.5, high=2.5, size=1 * N)).reshape(1, N))
        # 均匀分布 2-3 np.random.uniform   [N*1]
        g_i = np.mat(abs(np.random.uniform(low=2, high=3, size=1 * N)).reshape(1, N))

        # 完全本地
        local_time, local_lantency ,stop_time= EAOO_local(N, n, E_min, E_i, D_i_list, f_i, g_i, T_, Ps_)
        local_lantency_average = local_lantency / (stop_time + 1)
        local_lantency_list.append(local_lantency_average)
        local_time_list.append(local_time)

    print("local_lantency_list:",local_lantency_list)
    print("local_time_list",local_time_list)