import math
import scipy.io as sio  # import scipy.io for .mat file I/
from matplotlib.ticker import MultipleLocator
from memorySIC import MemoryDNN
from getfeasibleSIC_h import getfeasibleres
from GA import GA
import copy
from generate_h import *


def EAOO_latest_MAX(N_, n_, E_min_, P_, E_i_, D_i_list_, f_i_, g_i_, wirelessDevices_, server_, B_=5, T_=2.0, memory_=1024):
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


    # * 下面为一些参数，这些需要自己设置

    # N0 = 50  # 噪声功率
    alpha = 2  # 信号衰减系数
    beta = 0.5  # 阈值

    N = N_  # number of users
    n = n_  # number of time frames

    Delta = 32  # Update interval for adaptive K

    totallantency_final = 0  # * 初始化最终的总时延(3000个时间帧的)

    flagWD = [0 for fl in range(N)]  # 记录无线设备完成任务所需的时间帧数，即时延
    # print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d' % (N, n, K, decoder_mode, Memory, Delta))
    # Load data

    if N in [5, 6, 7, 8, 9, 10, 20, 30]:
        channel0 = sio.loadmat('./data/data_%d' % N)['input_h']
        # rate = sio.loadmat('./data/data_%d' % N)['output_obj']  # this rate is only used to plot figures; never used to train DROO.
    else:
        channel_temp = sio.loadmat('./data/data_%d' % 30)['input_h']
        # rate_temp = sio.loadmat('./data/data_%d' % 30)['output_obj']
        channel0 = channel_temp[:, 0:N]
        # rate = rate_temp[:, 0:N]
    # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
    channel = channel0 * 1000000
    # 根据设备位置新生成的信道增益
    channel_new0 = np.array(h_list)
    channel_new = channel_new0 * 1000000

    # split_idx = int(.8 * len(channel))
    split_idx = int(.8 * len(channel_new))
    # num_test = min(len(channel) - split_idx, n - int(.8 * n))  # training data size
    num_test = min(len(channel_new) - split_idx, n - int(.8 * n))  # training data size

    start_time = time.time()

    m_list_all = get_all_w(N_)
    latency_res_all = 0

    k_idx_his = []

    print('The algorithm EAOOSIC with', N, 'WDs begin.')

    P = P_[0, :].tolist()[0]
    f_i = f_i_[0, :].tolist()[0]
    E_min = E_min_[0, :].tolist()[0]
    E_i = E_i_[0, :].tolist()[0]
    g_i = g_i_[0, :].tolist()[0]


    stop_time = n - 1  # * 算法在哪个时间帧停止

    E_flag = False  # * 如果能力不足了就break


    # lowrate_ = lowrate   #*测试
    # print("本轮的minLocal Rate的最小值是：",lowrate_)
    # 遍历每一个时间帧
    for i in range(n):

        current_lot = i
        # print("第", current_lot, "个时间开始：")

        localsimple = 0  # 单个时间帧内，精简掉的设备的时延

        if i % (n // 10) == 0:
            pass
            # print("%0.1f" % (i / n))

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
        # E_min = E_min_[i, :].tolist()[0]
        D_i_list = D_i_list_[i, :].tolist()[0]

        T = T_  # 时间帧的长度
        B = B_  # 通信带宽
        N_0 = 1e-10  # 接收端噪声功率
        # 初始电量
        # 所有无线设备总剩余能量
        # Ps = Ps_  # 服务器P
        amin = 1e-27
        # 本地
        # 进行判断，数据上传时间大于一个时间帧，不适宜边缘计算

        C_local_ori = []
        C_up_ori = []
        uploadrecord_ori = []



        # print("时间帧",current_lot,"的f_i是:", f_i)

        # Q = []
        for fl in range(N):  # 上轮该无线设备未执行完任务，本轮不予分配新的任务
            if flagWD[fl] > T:        #第一个时间帧该设备需执行的时间 >T
                D_i_list[fl] = 0     #本时间帧 第二个时间帧 D_i = 0
                flagWD[fl] -= T  # 第一个时间帧后，还需要多久

        # local_lantency = 0  # 精简掉的设备的时延
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
                # pass
                # print(i)

        if E_flag:
            break

        # *对补全后的设备进行分组
        def split_group_latency(m):
            up_devices = []
            local_lantency = 0
            for i in range(len(m)):  # 先对所有设备进行筛选，m[i]=0的直接计算本地时延之和，m[i]=1的运用sic进行分组
                q_temp = 0
                if m[i] == 1:  # 边缘执行
                    up_devices.append(wireless_devices[i])
                else:  # 本地执行
                    if D_i_list[i] != 0:  # 上个时间帧没有执行完的设备
                        # q_temp = flagWD[i]
                        q_temp = D_i_list[i] * g_i[i] / f_i[i]
                    # print("每个设备的q_temp:",q_temp)
                    local_lantency += q_temp
            # split_list = sic(devices_all=up_devices, server=server, alpha=alpha, N0=N_0, beta=beta)
            split_list = sic_h(devices_all=up_devices, server=server, N0=N_0, beta=beta)
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


        latency_min_all = 99999999999
        final_m_all = []
        # * 计算 m_list_all 中的最佳情况
        for m in m_list_all:
            up_latency_all, local_latency_all = split_group_latency(m)
            # 各决策变量的总时延
            m_latency = up_latency_all + local_latency_all

            # 时延约束 和 能量约束 判断
            if up_latency_all <= T:
                # 能量约束判断
                for m_idx in range(len(m)):
                    energy_limit = E_i[m_idx] - ((1 - m[m_idx]) * C_local_ori[m_idx] + m[m_idx] * C_up_ori[m_idx])
                    if energy_limit < E_min[m_idx]:
                        break
                else:
                    if m_latency < latency_min_all:
                        latency_min_all = m_latency
                        final_m_all = m[:]
        # print("第 %d 个时间帧，当前的 遍历 结果为：%.4f" % (i, latency_min_all ))
        # print(" 遍历 ：  ", final_m_all)
        latency_res_all += latency_min_all


    total_time = time.time() - start_time

    return total_time, totallantency_final, stop_time,