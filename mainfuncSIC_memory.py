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
import itertools
import math
import scipy.io as sio  # import scipy.io for .mat file I/
import numpy as np  # import numpy
from matplotlib.ticker import MultipleLocator

from memorySIC import MemoryDNN
from getfeasibleSIC import getfeasibleres
import time
from GA import GA
import copy
from sic_compute import *


def plot_rate(rate_his, rolling_intv=50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)

    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))
    #    rolling_intv = 20

    plt.plot(np.arange(len(rate_array)) + 1, df.rolling(rolling_intv, min_periods=1).mean(), 'b')
    plt.fill_between(np.arange(len(rate_array)) + 1, df.rolling(rolling_intv, min_periods=1).min()[0],
                     df.rolling(rolling_intv, min_periods=1).max()[0], color='b', alpha=0.2)
    x_major_locator = MultipleLocator(1000)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.1)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(0, 10000)
    plt.ylim(0.4, 1)
    plt.ylabel('Normalized Time Delay')
    plt.xlabel('Time Frames')
    legend_font = {"family": "Times New Roman"}
    plt.legend(prop=legend_font)
    plt.show()


def EAOO_latest(N_, n_, E_min_, P_, E_i_, D_i_list_, f_i_, g_i_, B_=5, T_=2, Ps_=50, memory_=1024):
    wireless_devices, location = create_wireless_device(N_, 1, 1, 0.2, 0.35, 0.1)
    server = Server((location[0] + location[1]) / 2, (location[2] + location[3]) / 2)

    # 计算数据上传时延
    def dataUpload(B, P, h_i, N_0, D_i):  # 通信带宽，无线设备传输功率，信道增益，接收端噪声功率，无线设备完成任务需处理的数据量
        return D_i / (B * math.log2(1 + P * h_i / N_0))

    '''
    def helplamta(T, h_i, Ps, eh):
        return 1.0 * T * h_i * Ps * eh
    '''

    def localEnergyCost(a, f_i, D_i, g):
        return 1.0 * a * f_i * f_i * D_i * g

    # 计算数据上传能耗，返回是否小于最小能量
    def energyCost(E_i, C_up):
        return E_i - C_up

    def save_to_txt(rate_his, file_path):
        with open(file_path, 'w') as f:
            for rate in rate_his:
                f.write("%s \n" % rate)

    # * 下面为一些参数，这些需要自己设置

    # N0 = 50  # 噪声功率
    alpha = 2  # 信号衰减系数
    beta = 0.5  # 阈值

    N = N_  # number of users
    n = n_  # number of time frames
    K = N  # initialize K = N
    decoder_mode = 'OP'  # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = memory_  # capacity of memory structure
    Delta = 32  # Update interval for adaptive K

    totallantency_final = 0  # 初始化最终的总时延

    flagWD = [0 for fl in range(N)]  # 记录无线设备完成任务所需的时间帧数，即时延
    # print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d' % (N, n, K, decoder_mode, Memory, Delta))
    # Load data
    if N in [5, 6, 7, 8, 9, 10, 20, 30]:
        channel0 = sio.loadmat('./data/data_%d' % N)['input_h']
        rate = sio.loadmat('./data/data_%d' % N)[
            'output_obj']  # this rate is only used to plot figures; never used to train DROO.
    else:
        channel_temp = sio.loadmat('./data/data_%d' % 30)['input_h']
        rate_temp = sio.loadmat('./data/data_%d' % 30)['output_obj']
        channel0 = channel_temp[:, 0:N]
        rate = rate_temp[:, 0:N]
    # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
    channel = channel0 * 1000000

    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n))  # training data size
    lr = 1e-5  # 学习率
    mem = MemoryDNN(net=[N * 3, 120, 80, N],
                    learning_rate=lr,
                    training_interval=7,
                    batch_size=128,
                    memory_size=Memory
                    )
    ga = GA(N, s=[], x=None)
    start_time = time.time()

    rate_his = []
    rate_his_ratio = []
    # mode_his = []
    k_idx_his = []
    K_his = []
    print('The algorithm EAOOSIC with', N, 'WDs begin.')
    P = P_[0, :].tolist()
    E_i = E_i_[0, :].tolist()
    # eh_i = eh_i_[0, :].tolist()[0]
    # 计算速率
    f_i = f_i_[0, :].tolist()
    # CPU周期
    g_i = g_i_[0, :].tolist()
    E_min = E_min_[0, :].tolist()

    stop_time = n - 1  # * 算法在哪个时间帧停止

    E_flag = False  # * 如果能力不足了就break
    # 遍历每一个时间帧
    for i in range(n):

        current_lot = i
        # print("第", current_lot, "个时间开始：")

        localsimple = 0  # 单个时间帧内，精简掉的设备的时延

        if i % (n // 10) == 0:
            pass
            # print("%0.1f" % (i / n))
        if i > 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(k_idx_his[-Delta:-1]) + 1
            else:
                max_k = k_idx_his[-1] + 1
            K = min(max_k + 1, N)

        if i < n - num_test:
            # training
            i_idx = i % split_idx
        else:
            # test
            i_idx = i - n + num_test + split_idx

        h = channel[0, :]
        h0 = channel0[0, :]  # 判断是否超过一个时间帧用原始信道增益计算
        local_list = []  # 确定本地执行无线设备下标
        # E_min = E_min_[i, :].tolist()[0]

        T = T_  # 时间帧的长度
        B = B_  # 通信带宽
        N_0 = 1e-10  # 接收端噪声功率
        # 初始电量
        # 所有无线设备总剩余能量
        Ps = Ps_  # 服务器P
        amin = 1e-27
        # 本地
        # 进行判断，数据上传时间大于一个时间帧，不适宜边缘计算

        C_local_ori = []
        C_up_ori = []
        uploadrecord_ori = []

        D_i = []
        upload = []  # 记录决策变量精简后设备的上传时延
        recordD_i = []  # 记录决策变量精简后的处理任务量
        recordf_i = []  # 记录决策变量精简后的本地执行速率
        recordg_i = []  # 记录决策变量精简后执行任务所需cpucycle数
        edge_list = []  # 边缘执行的设备下标
        # Helplamta_rec = []  # 决策变量精简后设备采集到的能量
        E_i_record = []  # 决策变量精简后设备剩余能量
        C_up_rec = []
        C_local_rec = []
        E_min_rec = []
        q_list_local = []
        D_i_list = D_i_list_[i, :].tolist()
        # Q = []
        for fl in range(N):  # 上轮该无线设备未执行完任务，本轮不予分配新的任务
            if flagWD[fl] > T:
                D_i_list[fl] = 0
                flagWD[fl] -= T  # 本轮执行完后所需的时间帧数减一
        # tempupload = []

        # local_lantency = 0  # 精简掉的设备的时延

        for index in range(N):

            # 精简时w=1分析能量约束
            # H_get = getH(1, T, h0[index], Ps, eh_i[index])
            # help_lamta = helplamta(T, h0[index], Ps, eh_i[index])
            D_i.append(np.mean(D_i_list))
            C_up_E = P[index] * dataUpload(B, P[index], h0[index], N_0, D_i_list[index])
            C_up_ori.append(C_up_E)
            # 数据上传时延超出一个时间帧 或者 能耗超出当前电量
            uploadrecord = dataUpload(B, P[index], h0[index], N_0, D_i_list[index])  # 记录设备的数据上传时延

            uploadrecord_ori.append(uploadrecord)
            C_local = localEnergyCost(amin, f_i[index], D_i_list[index], g_i[index])
            C_local_ori.append(C_local)

            # ****此处判断EAOO算法是否因当前时间帧 设备的剩余能量  < 设备的最小剩余能量 而停止
            if E_i[index] < E_min[index]:
                stop_time = i
                stop_time = min(stop_time, i)  # 算法在第 i 个时间帧停止
                E_flag = True
                break;
                # pass
                # print(i)

            # 决策变量精简，确定为本地执行
            if uploadrecord > T \
                    or energyCost(E_i[index], C_up_E) < E_min[index]:
                local_list.append(index)
                # 上个时间帧内任务执行完，更新为该时间帧的时延；未执行完不变，表示剩余时延
                if D_i_list[index] != 0:
                    flagWD[index] = D_i_list[index] * g_i[index] / f_i[index]
                # 精简部分时延（确定本地执行）
                localsimple += D_i_list[index] * g_i[index] / f_i[index]
                if D_i_list[index] == 0:
                    q_temp = flagWD[index]
                else:
                    q_temp = 1. / (D_i_list[index] * g_i[index] / f_i[index])
                q_list_local.append(q_temp)
                # E_i[index] -= C_local
                continue
            else:
                # 记录决策变量精简后各设备的参数
                C_up_rec.append(C_up_E)
                C_local_rec.append(C_local)
                # Helplamta_rec.append(help_lamta)
                edge_list.append(index)
                E_i_record.append(E_i[index])
                E_min_rec.append(E_min[index])
                upload.append(uploadrecord)
                recordf_i.append(f_i[index])
                recordD_i.append(D_i_list[index])
                recordg_i.append(g_i[index])
            # E_i[index] -= C_up_E

        if E_flag:
            break

        N1 = N - len(local_list)  # 计算出N1 为确定服务器执行的数量
        # print("精简后的设备数量:",N1)
        mutation_pos = [n1 for n1 in range(N1)]
        # the action selection must be either 'OP' or 'KNN'
        # 传入三个参数信道增益 任务量 电量
        m_list = mem.decode(h, E_i, D_i_list, local_list, N1, decoder_mode)
        # print(N1)
        ga.s = copy.deepcopy(m_list)
        # print(ga.s)
        ga.pop_size = len(m_list)
        # print(i,edge_list,E_i)
        ga.crossAndMutation(mutation_pos)
        m_list.extend(ga.s)
        # print("m_list:",m_list)
        m_list_true = []  # 记录m_list中可行的卸载决策

        # 生成一组保底可行解
        feasible_decision = getfeasibleres(edge_list, upload, recordD_i, recordg_i, recordf_i, E_min_rec, C_local_rec,
                                           C_up_rec,
                                           E_i_record, wireless_devices, server, alpha, N_0, beta, T)
        # feasible_decision = getfeasibleres(edge_list, upload, recordD_i, recordg_i, recordf_i, E_min_rec, C_local_rec, C_up_rec,
        #         E_i, wireless_devices, server, alpha, N_0, beta, T)
        # print("原始生成的保底可行解:", feasible_decision)

        # 先补全这个保底的可行解
        for index in local_list:
            feasible_decision = np.insert(feasible_decision, index, 0)
            # feasible_decision[index] = 0
        # print("补全后的保底可行解:",feasible_decision)
        m_list_true.append(feasible_decision)

        # * 计算最大奖励
        r_list = []  # 记录时延---保底可行解和满足可行性分析的解
        # all_lantency_list = []  #记录时延---保底可行解和满足可行性分析的解
        # 计算保底可行解的时延
        feasible_lantency = 0
        for id in range(len(feasible_decision)):
            feasible_lantency += feasible_decision[id] * uploadrecord_ori[id] + (1 - feasible_decision[id]) * (
                    D_i[id] * g_i[id] / f_i[id])
        # all_lantency_list.append(feasible_lantency)
        r_list.append(feasible_lantency)


        # print("精简掉的设备的时延：", localsimple)

        # *对上传的设备进行分组
        def split_group_lantency(m):
            current_up = []  # 存的下标
            current_local = []  # 存的下标
            current_single = []  # 存的下标
            current_par = []  # 存的下标
            maxPar_lantency = []  # 存放每一组中时延最大的设备的所需时延  是时延

            # print("len(m)",len(m))
            # 将一一组卸载决策 m中的本地执行和边缘执行的设备分开
            for index in range(len(m)):
                if m[index] == 1:
                    current_up.append(index)
                else:
                    current_local.append(index)

            # 从wireless_devices中取current_up中对应下标的设备
            up_devices = []
            for index in current_up:
                up_devices.append(wireless_devices[index])
            split_list = sic(devices_all=up_devices, server=server, alpha=alpha, N0=N_0, beta=beta)

            # print("当前m的分组情况是：", split_list)
            # 分组情况是： [[9, 7, 6, 1, 8, 0], [4, 3, 2], [5]]
            # current_single[] and current_par

            for list in split_list:
                if len(list) != 1:
                    current_par.append(list)  # [[9, 7, 6, 1, 8, 0], [4, 3, 2]]
                else:
                    current_single.append(list)

            for par in current_par:  # par = [9, 7, 6, 1, 8, 0]
                par_lantency = []  # 存放每组设备上传时延
                for device in par:
                    par_lantency.append(dataUpload(B, P[device], h0[device], N_0, D_i_list[device]))
                maxPar_lantency.append(max(par_lantency))

            # 不能并行上传的设备的总时延   current_single = [[5]]
            single_lantency = 0
            for single_list in current_single:  # [5]
                for device in single_list:
                    single_lantency += dataUpload(B, P[device], h0[device], N_0, D_i_list[device])

            # 精简后仍本地执行的设备的总时延
            up_local_lantency = 0

            for device in current_local:
                up_local_lantency += recordD_i[device] * recordg_i[device] / recordf_i[device]
            return single_lantency, maxPar_lantency, up_local_lantency

        # m_listfull_lantency = []  # (不包含保底解的)存每个 可行解，补全后 设备的 总时延
        # * 可行性分析
        for m in m_list:

            single_lantency, maxPar_lantency, up_local_lantency = split_group_lantency(m)
            time_limit = sum(maxPar_lantency) + single_lantency

            # 各决策变量的总时延
            m_lantency = time_limit + up_local_lantency + localsimple

            # 补全每个决策变量 m
            for index in local_list:
                # m[index] = 0
                m = np.insert(m, index, 0)

            # 时延约束 和 能量约束 判断
            if time_limit <= T:
                # 能量约束判断
                for i in range(len(m)):
                    energy_limit = E_i[i] - ((1 - m[i]) * C_local_ori[i] + m[i] * C_up_ori[i])
                    if energy_limit < E_min[i]:
                        break
                else:
                    m_list_true.append(m)
                    # m_listfull_lantency.append(m_lantency)
                    # ll_lantency_list.append(m_lantency)
                    r_list.append(m_lantency)
        # print("可行解有：", m_list_true)
        # print("当前时间帧内，各可行解的总时延：",all_lantency_list)

        ##!!!m_list_true比total_simplelantency多一个元素：保底的那个

        '''
        # 计算其他可行解的时延
        for i in range(len(m_listfull_lantency)):
            r = m_listfull_lantency[i]
            r_list.append(r)
        # print("各个可行解的奖励：", r_list)
        '''
        final_m = m_list_true[np.argmin(r_list)]  # 从可行决策变量中选取时延最小的
        # totallantency_singleframe = all_lantency_list[np.argmax(r_list)]
        totallantency_singleframe = min(r_list)
        # print("第",current_lot,"个时间帧内最优总时延：",totallantency_singleframe)

        optimal_m = final_m  # 当前时间帧的最优解是optimal_m！！！
        # print("第",current_lot,"个时间帧,最优的可行解是：", optimal_m)
        # 未精简部分该时间帧是否执行完的更新

        # 3000个时间帧的总时延
        totallantency_final += totallantency_singleframe
        # print("到第", current_lot, "个时间帧为止，最优总时延是,", totallantency_final)

        # 将最优解中去除原来要精简掉的
        final_m = np.delete(final_m, local_list)

        # w = analysemiu(final_m, upload, T)

        for id in range(len(final_m)):
            if upload[id] != 0 and D_i_list[id] != 0:
                flagWD[edge_list[id]] = (final_m[id] * upload[id]) + (1 - final_m[id]) * (
                        recordD_i[id] * recordg_i[id] / recordf_i[id])
            # 加上精简之后的时延
            # totallantency_final += final_m[id] * upload[id] + (1 - final_m[id]) * ( recordD_i[id] * g_i[id] / f_i[id])

        # 各设备进行能量更新
        for index in range(N):
            if optimal_m[index] == 0:
                C_local = localEnergyCost(amin, f_i[index], D_i_list[index], g_i[index])
                E_i[index] -= C_local
            else:
                C_up_E = P[index] * dataUpload(B, P[index], h0[index], N_0, D_i_list[index])
                E_i[index] -= C_up_E

        # encode the mode with largest reward
        mem.encode(h, E_i, D_i_list, optimal_m)  # *w
        # the main code for DROO training ends here

        # the following codes store some interested metrics for illustrations
        # memorize the largest reward
        rate_his.append(np.max(r_list))
        rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
        # record the index of largest reward
        k_idx_his.append(np.argmax(r_list))
        # record K in case of adaptive K
        K_his.append(K)
        # mode_his.append(m_list[np.argmax(r_list)])

        #print("第 ", current_lot, "个时间帧为止，累计总时延是：", totallantency_final)

    total_time = time.time() - start_time
    memory_cost_list = []
    memory_cost = mem.memory_cost()
    #print(memory_cost)
    memory_cost_list.append(memory_cost)
    #plot_rate(rate_his_ratio)

    print(N, '个 WDs',"算法停止的时间帧(0-2999):", stop_time)

    # print(stop_time + 1, "个时间帧的总时延是：", totallantency_final)
    return total_time, totallantency_final, stop_time, memory_cost_list

