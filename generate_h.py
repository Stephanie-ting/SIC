from sic_compute import *
import numpy as np
import random

def cal_mean_h(devices_all: list, server: Server):
    A_d = 4.11
    d_e = 2.8
    f_c = 915e6
    h_mean_list = []

    for x_i in devices_all:
        d_i = ((x_i.x - server.x) ** 2 + (x_i.y - server.y) ** 2) ** 0.5
        print(i,'：di：',d_i)
        t = 3e8 / (4 * 3.14 * f_c * d_i)
        h_i = A_d * (t ** d_e)
        h_mean_list.append(h_i)

    return h_mean_list

def racian_mec(h,K):
    n = len(h)
    # K = 3 + 12 ** 0.5
    factor = K / (K + 1)
    t1 = np.random.randn(n)
    t2 = np.random.randn(n)

    # beta = sqrt(h * factor)  # LOSchannelamplitude
    # sigma = sqrt(h * (1 - factor) / 2)#scatteringsdv
    beta = []
    sigma = []
    x = []
    y = []
    g = []
    for i in range(n):
        beta_i = (h[i] * factor) ** 0.5
        beta.append(round(beta_i, 4))

        sigma_i = (h[i] * (1 - factor) / 2) ** 0.5
        sigma.append(sigma_i)

        x_i = sigma[i] * t1[i] + beta[i]
        x.append(x_i)
        y_i = sigma[i] * t2[i]
        y.append(y_i)
        g_i = x_i ** 2 + y_i ** 2
        g.append(g_i.tolist())

    return g


def generate_h(h_mean_list):
    N = 3000
    mean_h = h_mean_list
    k = 3 + 12 ** 0.5

    #3000个时间帧10个设备的
    h_list = []
    for i in range(N):
        h_t = racian_mec(mean_h,k)
        h_list.append(h_t)
    return h_list

