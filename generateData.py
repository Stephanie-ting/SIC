# coding=utf8
import scipy.io as sio
import numpy as np
print("hello wolrd!")
# 导入数据
for N in range(10, 32, 2):
    n = 3000
    # tips:固定成n个基础值 固定为10J
    E_min = np.mat(abs(np.random.uniform(low=10.0, high=20.0, size=n * N)).reshape(n, N))
    # 无线设备传输功率
    # tips:固定成n个基础值
    P = np.mat(abs(np.random.uniform(low=0.5, high=0.6, size=n * N)).reshape(n, N))
    # E_i = np.mat(abs(np.random.normal(loc=23.0, scale=5.0, size=n * N)).reshape(n, N))
    # tips:固定成n个基础值 初始电量[N*1]
    E_i = np.mat(abs(np.random.uniform(low=500.0, high=600.0, size=n * N)).reshape(n, N))
    # 任务数据量 均匀分布 50-100 [N*n]
    D_i_list = np.mat(abs(np.random.uniform(low=100, high=150, size=n * N)).reshape(n, N))
    # 计算速率 均匀分布 50-100 [N*1]
    # f_i = np.mat(abs(np.random.uniform(low=80, high=100, size=1 * N)).reshape(1, N))
    f_i = np.mat(abs(np.random.uniform(low=50, high=150, size=n * N)).reshape(n, N))
    # CPU周期 [N*1]
    # g_i = np.mat(abs(np.random.uniform(low=1.5, high=2.5, size=1 * N)).reshape(1, N))
    # 均匀分布 2-3 np.random.uniform   [N*1]
    g_i = np.mat(abs(np.random.uniform(low=2, high=3, size=n * N)).reshape(n, N))
    print("正在生成设备数为", N,"的数据:")
    sio.savemat('./data/myData_'+str(N) + '.mat',{'E_min': E_min,'P':P,'E_i':E_i,'D_i_list':D_i_list,'f_i':f_i,'g_i':g_i})

