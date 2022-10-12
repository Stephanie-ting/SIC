import itertools
import math
import scipy.io as sio  # import scipy.io for .mat file I/
import numpy as np  # import numpy
from matplotlib.ticker import MultipleLocator

from memorySIC import MemoryDNN
from getfeasible import getfeasibleres
import time
from GA import GA
import copy
from generate_h import *



def EAOO_latest_serial(N_, n_, E_min_, P_, E_i_, D_i_list_, f_i_, g_i_, wirelessDevices_, server_,B_=5, T_=2,N_0_=1e-10):
    # wireless_devices, location = create_wireless_device(N_, 1, 1, 0.2, 0.35, 0.1)
    # server = Server((location[0] + location[1]) / 2, (location[2] + location[3]) / 2)
    wireless_devices = wirelessDevices_
    server = server_
    h_mean_list = cal_mean_h(devices_all=wireless_devices, server=server)
    h_list = generate_h(h_mean_list)

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

    # 便于判断  所有上传任务的  时间约束
    def analysemiu(m, upload, T):
        tempres = []
        for i in range(len(m)):
            tempres.append(m[i] * upload[i])
        return (T - sum(tempres))

    # * 下面为一些参数，这些需要自己设置

    # N0 = 50  # 噪声功率
    alpha = 2  # 信号衰减系数
    beta = 0.5  # 阈值

    N = N_  # number of users
    n = n_  # number of time frames
    K = N  # initialize K = N
    decoder_mode = 'OP'  # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = 512  # capacity of memory structure
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

    # 根据设备位置新生成的信道增益
    channel_new0 = np.array(h_list)
    channel_new = channel_new0 * 1000000

    # split_idx = int(.8 * len(channel))
    split_idx = int(.8 * len(channel_new))
    # num_test = min(len(channel) - split_idx, n - int(.8 * n))  # training data size
    num_test = min(len(channel_new) - split_idx, n - int(.8 * n))  # training data size
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
    print('The algorithm EAOO with', N, 'WDs begin.')

    P = P_[0, :].tolist()[0]
    f_i = f_i_[0, :].tolist()[0]
    E_min = E_min_[0, :].tolist()[0]
    E_i = E_i_[0, :].tolist()[0]
    # CPU周期
    g_i = g_i_[0, :].tolist()[0]

    stop_time = n - 1  # * 算法在哪个时间帧停止

    E_flag = False  # * 如果能力不足了就break

    # lowrate_ = lowrate  #*测试local rate
    # print("本轮的minLocal Rate的最小值是：", lowrate_)
    # 遍历每一个时间帧
    for i in range(n):

        # f_i = f_i_[i, :].tolist()[0]
        current_lot = i
        # print("第", current_lot, "个时间开始：")

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

        h_ori = channel[0, :]
        h = channel_new[0, :]
        # print('channel0\n',channel0.shape)
        h0_ori = channel0[0, :]  # 判断是否超过一个时间帧用原始信道增益计算
        # print('h0ori\n', h0_ori)
        h0 = channel_new0[0, :]  # 判断是否超过一个时间帧用原始信道增益计算
        # print('h0new\n', h0)
        local_list = []  # 确定本地执行无线设备下标
        # E_min = E_min_[i, :].tolist()[0]

        T = T_  # 时间帧的长度
        B = B_  # 通信带宽
        # N_0 = 1e-10  # 接收端噪声功率
        N_0 = N_0_
        amin = 1e-27

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

        D_i_list = D_i_list_[i, :].tolist()[0]

        # print("时间帧", current_lot, "的f_i是:", f_i)

        # Q = []
        for fl in range(N):  # 上轮该无线设备未执行完任务，本轮不予分配新的任务
            if flagWD[fl] > T:
                D_i_list[fl] = 0
                flagWD[fl] -= T  # 本轮执行完后所需的时间帧数减一
        # tempupload = []

        for index in range(N):

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
                break

            # 决策变量精简，确定为本地执行
            if uploadrecord > T \
                    or energyCost(E_i[index], C_up_E) < E_min[index]:
                local_list.append(index)
                # 上个时间帧内任务执行完，更新为该时间帧的时延；未执行完不变，表示剩余时延
                if D_i_list[index] != 0:
                    flagWD[index] = D_i_list[index] * g_i[index] / f_i[index]
                # # 精简部分时延（确定本地执行）
                # totallantency_final += D_i_list[index] * g_i[index] / f_i[index]
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
                                           C_up_rec, E_i_record, T)
        # feasible_decision = getfeasibleres(edge_list, upload, recordD_i, recordg_i, recordf_i, E_min_rec, C_local_rec, C_up_rec,
        #         E_i, wireless_devices, server, alpha, N_0, beta, T)
        # print("原始生成的保底可行解:", feasible_decision)

        # 先补全这个保底的可行解
        for index in local_list:
            feasible_decision = np.insert(feasible_decision, index, 0)
            # feasible_decision[index] = 0
        # print("补全后的保底可行解:",feasible_decision)
        m_list_true.append(feasible_decision)

        # * 计算最大奖励----最小时延
        r_list = []
        # 计算保底可行解的时延
        feasible_lantency = 0
        for id in range(len(feasible_decision)):
            feasible_lantency += feasible_decision[id] * uploadrecord_ori[id] + (1 - feasible_decision[id]) * (
                        D_i_list[id] * g_i[id] / f_i[id])
        r_list.append(feasible_lantency)

        # 先补全每个决策变量 m
        for j in range(len(m_list)):
            for index in local_list:
                m_list[j] = np.insert(m_list[j], index, 0)

        # m_list_lantency = []
        # * 可行性分析
        for m in m_list:
            # 补全变量后，单个决策变量的总时延
            m_lantency = 0

            # print("此时的m：",m)
            for id in range(len(m)):
                m_temp = 0
                # time_limit  += m[id] * uploadrecord_ori[id]
                if D_i_list[id] != 0:
                    # m_temp = flagWD[id]
                    m_temp = m[id] * uploadrecord_ori[id] + (1 - m[id]) * (D_i_list[id] * g_i[id] / f_i[id])
                # print("每个设备的m_temp:",m_temp)
                m_lantency += m_temp
            # print("m_lantecy:",m_lantency)

            # 时延约束 和 能量约束 判断
            if analysemiu(m, uploadrecord_ori, T) >= 0:
                # 能量约束判断
                for i in range(len(m)):
                    energy_limit = E_i[i] - ((1 - m[i]) * C_local_ori[i] + m[i] * C_up_ori[i])
                    if energy_limit < E_min[i]:
                        break
                else:
                    m_list_true.append(m.tolist())
                    r_list.append(m_lantency)

            # # 能量约束计算
            # energy_limit = 0
            # for i in range(len(m)):
            #     energy_limit = E_i[i] - ((1 - m[i]) * C_local_ori[i] + m[i] * C_up_ori[i])
            #
            # # 约束条件判断
            # if (energy_limit >= E_min[i]) and (analysemiu(m, uploadrecord_ori, T) >= 0):
            #     m_list_true.append(m.tolist())
            #     r_list.append(m_lantency)
        # print("可行解有：", m_list_true)

        final_m = m_list_true[np.argmin(r_list)]  # 从可行决策变量中选取时延最小的
        # print("第", current_lot, "个时间帧内r_list：", r_list)
        totallantency_singleframe = min(r_list)
        # print("第",current_lot,"个时间帧内最优总时延：",totallantency_singleframe)

        optimal_m = final_m  # 最优解是optimal_m！！！
        # print("第",current_lot,"个时间帧,最优的可行解是：", optimal_m)

        # 3000个时间帧的总时延
        totallantency_final += totallantency_singleframe

        # for index in local_list:
        final_m = np.delete(final_m, local_list)

        # w = analysemiu(final_m, upload, T)

        # 未精简部分该时间帧是否执行完的更新
        for id in range(len(final_m)):
            if upload[id] != 0 and D_i_list[id] != 0:
                flagWD[edge_list[id]] = (final_m[id] * upload[id]) + (1 - final_m[id]) * (
                            recordD_i[id] * recordg_i[id] / recordf_i[id])
            # 未精简部分时延
            # totallantency_final += final_m[id] * upload[id] + (1 - final_m[id]) * (recordD_i[id] * recordg_i[id] / recordf_i[id])

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
        rate_his.append(np.min(r_list))
        rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
        # record the index of largest reward
        k_idx_his.append(np.argmin(r_list))
        # record K in case of adaptive K
        K_his.append(K)
        # mode_his.append(m_list[np.argmax(r_list)])

        # print("第 ",current_lot,"个时间帧为止，累计总时延是：",totallantency_final)

    total_time = time.time() - start_time
    # mem.plot_cost()
    # plot_rate(rate_his_ratio)
    #
    print(N, '个 WDs', "算法停止的时间帧(0-2999):", stop_time)
    # print("本轮（3000个时间帧），最优总时延是,", totallantency_final)
    # print("本轮（3000个时间帧）为止，单个时间帧最优总时延平均值是,", totallantency_final/stop_time)

    # print(n, "个时间帧的总时延是：", totallantency_final)
    return total_time, totallantency_final, stop_time


def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)


if __name__ == "__main__":

    EAOO_lantency_list = []
    EAOO_time_list = []

    B_ = 30
    T_ = 2
    # Ps_ = 50
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

        # EAOO-串行
        EAOO_time, EAOO_lantency, stop_time = EAOO_latest_serial(N, n, E_min, P, E_i, D_i_list, f_i, g_i, B_, T_)
        EAOO_lantency_average = EAOO_lantency / (stop_time + 1)
        EAOO_lantency_list.append(EAOO_lantency_average)
        EAOO_time_list.append(EAOO_time)

    print('------EAOO---------')
    print("EAOO_time of  each 10-30 devices", EAOO_time_list)
    print("EAOO_lantency_average of each 10-30 devices,", EAOO_lantency_list)

    save_to_txt(EAOO_time_list, "EAOO_time.txt")
    save_to_txt(EAOO_lantency_list, "EAOO_lantency.txt")
