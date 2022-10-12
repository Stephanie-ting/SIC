from sic_compute import *
import numpy as np


def cal_mean_h(devices_all: list, server: Server):
    A_d = 4.11  #天线增益
    d_e = 2.8  #路径损失指数
    f_c = 915e6
    h_mean_list = []

    for x_i in devices_all:
        d_i = ((x_i.x - server.x) ** 2 + (x_i.y - server.y) ** 2) ** 0.5
        # print(i,'：di：',d_i)
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

if __name__ == "__main__":

    # n为无线设备数目
    n = 20
    # * 下面为一些参数，这些需要自己设置
    N0 = 50  # 噪声功率
    alpha = 2  # 信号衰减系数
    beta = 0.5

    figure, axes = plt.subplots()
    wireless_devices, location = create_wireless_device(n, 1, 1, 0.2, 0.35, 0.1)
    print("wireless_devices", wireless_devices)
    i = 1
    for device in wireless_devices:
        print("第%d个无线设备的位置是： x: %f,  y: %f,  r: %f" % (i, device.x, device.y, device.r))
        draw_circle = plt.Circle((device.x, device.y), device.r, color='b', fill=False)
        axes.add_artist(draw_circle)
        i += 1
    print(location)
    server = Server((location[0] + location[1]) / 2, (location[2] + location[3]) / 2)
    plt.scatter(server.x, server.y, s=10)
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    plt.show()
    start_time = time.time()
    split_list = sic(devices_all=wireless_devices, server=server, alpha=alpha, N0=N0, beta=beta)
    split_list_h = sic_h(devices_all=wireless_devices, server=server, alpha=alpha, N0=N0, beta=beta)

    print("sic分组情况是：", split_list)
    print("sic_h分组情况是：", split_list_h)
    # print("最大并发度为：", res)
    end_time = time.time()
    # print("总时间为：", end_time - start_time)

    h_mean_list = cal_mean_h(devices_all=wireless_devices, server=server)
    print('h_i:',h_mean_list)
    #h_mean = [2.09586234e-06, 2.78914438e-06, 3.25940045e-06, 2.66778002e-06, 2.05608874e-06, 2.32011316e-06, 2.19657710e-06, 2.47194857e-06, 2.75105669e-06, 2.10990325e-06]

    h_list = generate_h(h_mean_list)
    print('h:',h_list)



