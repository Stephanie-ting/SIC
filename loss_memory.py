import numpy as np
import scipy.io as sio
from mainfuncSIC_memory import EAOO_latest



def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == '__main__':


    B_ = 30
    T_ = 2
    Ps_ = 50
    memory = [128, 256, 512, 1024, 2048]
    for i in range(len(memory)):
        memory_ = memory[i]
        N = 10
        n = 3000

        E_min = sio.loadmat('./data/myData_%d' % N)['E_min']
        P = sio.loadmat('./data/myData_%d' % N)['P']
        E_i = sio.loadmat('./data/myData_%d' % N)['E_i']
        D_i_list = sio.loadmat('./data/myData_%d' % N)['D_i_list']
        f_i = sio.loadmat('./data/myData_%d' % N)['f_i']
        g_i = sio.loadmat('./data/myData_%d' % N)['g_i']

        # EAOO-SIC算法
        EAOOSIC_time, EAOOSIC_lantency, stop_time_sic,memory_cost_list = EAOO_latest(N, n, E_min, P, E_i, D_i_list, f_i, g_i, B_, T_,Ps_,memory_)


        sio.savemat('./memory/cost_EAOOSIC_' + str(memory[i]) + '.mat',{'memory_cost_list':memory_cost_list})


