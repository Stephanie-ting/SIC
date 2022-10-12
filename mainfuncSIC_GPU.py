# coding=utf8
import math
import scipy.io as sio  # import scipy.io for .mat file I/
import numpy as np  # import numpy
from matplotlib.ticker import MultipleLocator
from memory_GPU import MemoryDNN
from getfeasibleSIC_h import getfeasibleres
from GA import GA
import copy
from generate_h import *

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



def EAOO_latest_GPU(N_, n_, E_min_, P_, E_i_, D_i_list_, f_i_, g_i_,wirelessDevices_, server_,B_=5, T_=2):
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

    def save_to_txt(rate_his, file_path):
        with open(file_path, 'w') as f:
            for rate in rate_his:
                f.write("%s \n" % rate)

    def minMaxScale(data: np.array) -> np.array:
        Min = np.min(data)
        Max = np.max(data)
        return (data - Min) / (Max - Min)

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

    totallantency_final = 0  # * 初始化最终的总时延(3000个时间帧的)

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
    # print('channel0:\n',channel0)
    # print('channel:\n', channel)

    #根据设备位置新生成的信道增益
    channel_new0 = np.array(h_list)
    channel_new = channel_new0 * 1000000

    # split_idx = int(.8 * len(channel))
    split_idx = int(.8 * len(channel_new ))
    # num_test = min(len(channel) - split_idx, n - int(.8 * n))  # training data size
    num_test = min(len(channel_new) - split_idx, n - int(.8 * n))  # training data size
    lr = 1e-5  # 学习率
    mem = MemoryDNN(net=[N * 3, 300, 200, N],
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

    P = P_[0, :].tolist()[0]

    E_min = E_min_[0, :].tolist()[0]
    E_i = E_i_[0, :].tolist()[0]
    g_i = g_i_[0, :].tolist()[0]
    f_i = f_i_[0, :].tolist()[0]

    # print('原始生成的f_i_：',f_i_)
    # print('原始生成的E_i_：', E_i_)

    stop_time = n - 1  # * 算法在哪个时间帧停止

    E_flag = False  # * 如果能力不足了就break

    ## 最大最小值归一化以获得更好的训练
    E_i_list_scaled = minMaxScale(E_i_)
    # h_list_scaled = minMaxScale(channel)
    h_list_scaled = minMaxScale(channel_new)
    D_i_list_scaled = minMaxScale(D_i_list_)

    E_i_scaled = E_i_list_scaled[0, :].tolist()[0]

    # 遍历每一个时间帧
    for i in range(n):

        current_lot = i
        # print("第", current_lot, "个时间开始：").\
        # f_i = f_i_[i, :].tolist()[0]
        # print("时间帧", current_lot, "的f_i是:", f_i)
        # print("时间帧", current_lot, "的E_i是:", E_i)

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
        # print('h0new\n',h0)

        local_list = []  # 确定本地执行无线设备下标


        D_i_list = D_i_list_[i, :].tolist()[0]
        # print("时间帧", current_lot, "的初始分配D_i_list是:", D_i_list)

        # 获取输入数据的归一化值，用于神经网络
        D_i_scaled = D_i_list_scaled[i, :].tolist()[0]
        h_scaled = h_list_scaled[0, :]

        T = T_  # 时间帧的长度
        B = B_  # 通信带宽
        N_0 = 1e-10  # 接收端噪声功率
        amin = 1e-27
        C_local_ori = []
        C_up_ori = []
        uploadrecord_ori = []

        # D_i = []
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



        # Q = []
        for fl in range(N):  # 上轮该无线设备未执行完任务，本轮不予分配新的任务
            if flagWD[fl] > T:        #第一个时间帧该设备需执行的时间 >T
                D_i_list[fl] = 0     #本时间帧 第二个时间帧 D_i = 0
                flagWD[fl] -= T  # 第一个时间帧后，还需要多久

        # print("时间帧", current_lot, "的实际的D_i_list是:", D_i_list)

        # local_lantency = 0  # 精简掉的设备的时延
        for index in range(N):

            C_up_E = P[index] * dataUpload(B, P[index], h0[index], N_0, D_i_list[index])
            C_up_ori.append(C_up_E)
            # 数据上传时延超出一个时间帧 或者 能耗超出当前电量
            uploadrecord = dataUpload(B, P[index], h0[index], N_0, D_i_list[index])  # 记录设备的数据上传时延

            uploadrecord_ori.append(uploadrecord)
            C_local = localEnergyCost(amin, f_i[index], D_i_list[index], g_i[index])
            C_local_ori.append(C_local)

            # ****此处判断SIC算法是否因当前时间帧 设备的剩余能量  < 设备的最小剩余能量 而停止
            if E_i[index] < E_min[index]:
                stop_time = i
                stop_time = min(stop_time, i)  # 算法在第 i 个时间帧停止
                E_flag = True
                break
                # pass
                # print(i)

            # 决策变量精简，确定为本地执行
            if uploadrecord > T or energyCost(E_i[index], C_up_E) < E_min[index]:
                local_list.append(index)
                # 上个时间帧内任务执行完，更新为该时间帧的时延；未执行完不变，表示剩余时延
                if D_i_list[index] != 0:    #上个时间帧内任务执行完，更新为该时间帧的时延
                    flagWD[index] = D_i_list[index] * g_i[index] / f_i[index]
            else:
                # 记录决策变量精简后各设备的参数
                C_up_rec.append(C_up_E)
                C_local_rec.append(C_local)
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
        m_list_true = []  # 记录保底可行解 + m_list中可行的卸载决策


        # 生成一组保底可行解
        feasible_decision = getfeasibleres(edge_list, upload, recordD_i, recordg_i, recordf_i, E_min_rec, C_local_rec,
                                           C_up_rec,
                                           E_i_record, wireless_devices, server,  N_0, beta, T)
        # print("原始生成的保底可行解:", feasible_decision)

        # 先补全这个保底的可行解
        for index in local_list:
            feasible_decision = np.insert(feasible_decision, index, 0)
        # print("补全后的保底可行解:", feasible_decision)
        m_list_true.append(feasible_decision)

        # 先补全每个决策变量 m
        for j in range(len(m_list)):
            for index in local_list:
                m_list[j] = np.insert(m_list[j], index, 0)
            # print("补全后的m：", m_list[j])

        # *对补全后的设备进行分组
        def split_group_latency(m):
            up_devices = []
            local_lantency = 0
            for i in range(len(m)):  # 先对所有设备进行筛选，m[i]=0的直接计算本地时延之和，m[i]=1的运用sic进行分组
                q_temp = 0
                if m[i] == 1:   #边缘执行
                    up_devices.append(wireless_devices[i])
                else:  #本地执行
                    if D_i_list[i] != 0:    #上个时间帧没有执行完的设备
                        # q_temp = flagWD[i]
                        q_temp = D_i_list[i] * g_i[i] / f_i[i]
                    # print("每个设备的q_temp:",q_temp)
                    local_lantency += q_temp
            # split_list = sic(devices_all=up_devices, server=server, alpha=alpha, N0=N_0, beta=beta)
            split_list = sic_h(devices_all=up_devices, server=server,  N0=N_0, beta=beta)
            # print('分组情况:',split_list)

            up_lantency = 0
            for lst in split_list:  # [[1,2,3],[4,6],[7]]
                up_time = 0  # 每个分组中最大的上传时延
                for id in lst:  # [1,2,3]
                    up_time_temp = 0
                    if D_i_list[id] != 0:
                        up_time_temp = uploadrecord_ori[id]
                    up_time = max(up_time, up_time_temp)
                up_lantency += up_time  # 所有的组的最大的上传时延之和

            return up_lantency, local_lantency

        # * 计算最大奖励
        r_list = []  # 记录时延---保底可行解和满足可行性分析的解
        feasible_up, feasible_local = split_group_latency(feasible_decision)
        feasible_lantency = feasible_up + feasible_local
        r_list.append(feasible_lantency)
        # print("保底时延：",feasible_lantency)

        # * 可行性分析
        for m in m_list:
            up_lantency, local_lantency = split_group_latency(m)

            # 各决策变量的总时延
            m_lantency = up_lantency + local_lantency

            # 时延约束 和 能量约束 判断
            if up_lantency <= T:
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
            # #约束条件判断
            # if (energy_limit >= E_min[i]) and (up_lantency <= T):
            #     m_list_true.append(m.tolist())
            #     r_list.append(m_lantency)


        # print("可行解有：", m_list_true)

        final_m = m_list_true[np.argmin(r_list)]  # 从可行决策变量中选取时延最小的决策
        # totallantency_singleframe = all_lantency_list[np.argmax(r_list)]
        totallantency_singleframe = min(r_list)
        # print("第",current_lot,"个时间帧内最优总时延：",totallantency_singleframe)

        optimal_m = final_m  # 当前时间帧的最优解是optimal_m！！！
        # print("第", current_lot, "个时间帧,最优的可行解是：", optimal_m)

        # 3000个时间帧的总时延
        totallantency_final += totallantency_singleframe

        # 将最优解中去除原来要精简掉的
        final_m = np.delete(final_m, local_list)

        # 未精简部分该时间帧是否执行完的更新
        for id in range(len(final_m)):
            if upload[id] != 0 and D_i_list[id] != 0:
                flagWD[edge_list[id]] = (final_m[id] * upload[id]) + (1 - final_m[id]) * ( recordD_i[id] * recordg_i[id] / recordf_i[id])

        # 各设备进行能量更新
        for index in range(N):
            if optimal_m[index] == 0:
                C_local = localEnergyCost(amin, f_i[index], D_i_list[index], g_i[index])
                E_i[index] -= C_local
            else:
                C_up_E = P[index] * dataUpload(B, P[index], h0[index], N_0, D_i_list[index])
                E_i[index] -= C_up_E

        # encode the mode with largest reward
        # mem.encode(h_scaled, E_i_scaled, D_i_scaled, optimal_m)  # *w
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

    total_time = time.time() - start_time
    # mem.plot_cost()
    # plot_rate(rate_his_ratio)

    # print(N, '个 WDs', "算法停止的时间帧(0-2999):", stop_time)
    # print("本轮（3000个时间帧），最优总时延是,", totallantency_final)
    # print("本轮（3000个时间帧）为止，单个时间帧最优总时延平均值是,", totallantency_final/stop_time)
    print(N, '个 WDs',"算法停止的时间帧(0-2999):", stop_time)

    # print(stop_time + 1, "个时间帧的总时延是：", totallantency_final)
    return total_time, totallantency_final, stop_time