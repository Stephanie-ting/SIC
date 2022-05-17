# coding=utf8
import scipy.io as sio
from mainfuncSIC import EAOO_latest


def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)


if __name__ == "__main__":
    EAOOSIC_lantency_list_CPU = []
    EAOOSIC_time_list_CPU = []

    B_ = 30
    T_ = 2
    Ps_ = 50
    for N in range(10, 32, 2):
        n = 3000
        # tips:固定成n个基础值 固定为10J

        E_min = sio.loadmat('./data/myData_%d' % N)['E_min']
        P = sio.loadmat('./data/myData_%d' % N)['P']
        E_i = sio.loadmat('./data/myData_%d' % N)['E_i']
        D_i_list = sio.loadmat('./data/myData_%d' % N)['D_i_list']
        f_i = sio.loadmat('./data/myData_%d' % N)['f_i']
        g_i = sio.loadmat('./data/myData_%d' % N)['g_i']

        # EAOO-SIC算法
        EAOOSIC_time_CPU, EAOOSIC_lantency, stop_time_sic = EAOO_latest(N, n, E_min, P, E_i, D_i_list, f_i, g_i, B_, T_, Ps_)
        EAOOSIC_lantency_average = EAOOSIC_lantency / (stop_time_sic + 1)  # * 获得具体停止的时间帧stop_time，根据改时间帧得到平均lantency
        EAOOSIC_lantency_list_CPU.append(EAOOSIC_lantency_average)
        EAOOSIC_time_list_CPU.append(EAOOSIC_time_CPU)

    print('------EAOO-SIC-CPU-----')
    print("EAOOSIC_time_CPU of  each 10-30 devices", EAOOSIC_time_list_CPU)
    print("EAOOsic_lantency_average_CPU of each 10-30 devices,", EAOOSIC_lantency_list_CPU)

    save_to_txt(EAOOSIC_time_list_CPU, "EAOOSIC_time_CPU.txt")
    save_to_txt(EAOOSIC_lantency_list_CPU, "EAOOSIC_lantency_CPU.txt")



