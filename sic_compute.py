import random
import time
from typing import List, Any
import matplotlib.pyplot as plt
# from generate_h import *

class WirelessDevice:
    def __init__(self, x, y, r, number=-1) -> None:
        self.power = 50
        self.x = x
        self.y = y
        self.r = r
        self.number = number


class Server:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y


def output(devices_with_power: list):
    devices = []
    for x in devices_with_power:
        devices.append(x[0].number)
    return devices

# (n, 1, 1, 0.2, 0.35, 0.1)
def create_wireless_device(n: int, max_location_x: float, max_location_y: float,
                           min_r: float, max_r: float, min_distance: float) -> tuple:
    wireless_devices = []
    # * location从0到3分别表示的：左边界，右边界，下边界，上边界
    location = [-10, 20, -10, 20]
    while len(wireless_devices) < n:
        temp_x = random.uniform(0, max_location_x)
        temp_y = random.uniform(0, max_location_y)
        temp_r = random.uniform(min_r, max_r)
        #每个策略选出一个服务器位置，判断是否满足传输范围条件
        #贪婪：三个交叉点&中心点的并发度最大值
        for device in wireless_devices:
            if (device.x - temp_x) ** 2 + (device.y - temp_y) ** 2 >= (device.r + temp_r - min_distance) ** 2:
                break
        else:
            wireless_devices.append(WirelessDevice(temp_x, temp_y, temp_r, len(wireless_devices)))
            location[0] = max(location[0], temp_x - temp_r)
            location[1] = min(location[1], temp_x + temp_r)
            location[2] = max(location[2], temp_y - temp_r)
            location[3] = min(location[3], temp_y + temp_r)
    return wireless_devices, location


def sic(devices_all: list, server: Server, alpha: float, N0: float, beta: float) -> List[List[Any]]:
    # devices_with_power用于保存p/d^alpha
    devices_with_power = []
    for x in devices_all:
        distance = ((x.x - server.x) ** 2 + (x.y - server.y) ** 2) ** 0.5
        # 将设备和p/d^alpha以tuple的形式存入到数组中
        devices_with_power.append((x, x.power / distance ** alpha))
    devices_with_power.sort(key=lambda x: -x[1])

    # 保存有没有被遍历到
    record_temp = [False] * len(devices_with_power)
    res = []
    for k in range(len(devices_with_power)):
        if record_temp[k]:
            continue
        temp = []  # 一个分组
        for i in range(k, len(devices_with_power)):
            if record_temp[i]:
                continue
            if len(temp) == 0:
                temp.append(devices_with_power[i])
                record_temp[i] = True
            else:
                temp.append(devices_with_power[i])
                if sic_help(devices_with_power=temp,N0=N0, beta=beta):
                    record_temp[i] = True
                else:
                    temp.pop()

        if len(temp) > 0:
            res.append(temp)

    split_list = []  # 记录分组情况，存放设备下标index
    for group in res:  # group :[(,),(,),(,)]
        group_list = []
        for group_dev in group:  # group_dev:(x,with power)
            group_list.append(group_dev[0].number)
            # print(group_dev[0].number)
        split_list.append(group_list)
        # print("-----------")

    # print("分组情况是：", split_list)
    # return res
    return split_list


# 判断在当前无线设备组下是否能够解码（当前无限设备组的p/d^alpha已经排序）
def sic_help(devices_with_power: list,  N0: float, beta: float) -> bool:
    # 获得总共p/d^alpha之和
    total_power = 0
    for x in devices_with_power:
        total_power += x[1]

    # 开始判断当前设备组能否全部解码
    for x in devices_with_power:
        temp = x[1]
        if temp / (total_power - temp + N0) >= beta:
            total_power -= temp
        else:
            return False
    return True


def get_all_w(n: int) -> list:
    res = []

    def get_all_w_helper(curr: list):
        if len(curr) == n:
            res.append(curr.copy())
            return
        curr.append(0)
        get_all_w_helper(curr)

        curr[len(curr) - 1] = 1
        get_all_w_helper(curr)

        curr.pop()

    get_all_w_helper([])
    return res

def sic_h(devices_all: list, server: Server, N0: float, beta: float) -> List[List[Any]]:
    #h_mean
    A_d = 4.11  #天线增益
    d_e = 2.8  #路径损失指数
    f_c = 915e6  #载波频率
    h_mean_list = []

    # devices_with_power用于保存p * h
    devices_with_power = []
    for x in devices_all:
        distance = ((x.x - server.x) ** 2 + (x.y - server.y) ** 2) ** 0.5

        #计算每个h
        t = 3e8 / (4 * 3.14 * f_c * distance)
        h_i = A_d * (t ** d_e)
        h_mean_list.append(h_i)
        # 将设备和 p * h 以tuple的形式存入到数组中
        devices_with_power.append((x, x.power * h_i))
    devices_with_power.sort(key=lambda x: -x[1])

    # 保存有没有被遍历到
    record_temp = [False] * len(devices_with_power)
    res = []
    for k in range(len(devices_with_power)):
        if record_temp[k]:
            continue
        temp = []  # 一个分组
        for i in range(k, len(devices_with_power)):
            if record_temp[i]:
                continue
            if len(temp) == 0:
                temp.append(devices_with_power[i])
                record_temp[i] = True
            else:
                temp.append(devices_with_power[i])
                if sic_help(devices_with_power=temp,  N0=N0, beta=beta):
                    record_temp[i] = True
                else:
                    temp.pop()

        if len(temp) > 0:
            res.append(temp)

    split_list = []  # 记录分组情况，存放设备下标index
    for group in res:  # group :[(,),(,),(,)]
        group_list = []
        for group_dev in group:  # group_dev:(x,with power)
            group_list.append(group_dev[0].number)
            # print(group_dev[0].number)
        split_list.append(group_list)
        # print("-----------")

    # print("分组情况是：", split_list)
    # return res
    return split_list

if __name__ == "__main__":

    # n为无线设备数目
    n = 10
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
    split_list_h = sic_h(devices_all=wireless_devices, server=server, N0=N0, beta=beta)

    print("sic分组情况是：", split_list)
    print("sic_h分组情况是：", split_list_h)
    # print("最大并发度为：", res)
    end_time = time.time()
    # print("总时间为：", end_time - start_time)

    # h_mean_list = cal_mean_h(devices_all=wireless_devices, server=server)
    # print('h_i:',h_mean_list)
    #h_mean = [2.09586234e-06, 2.78914438e-06, 3.25940045e-06, 2.66778002e-06, 2.05608874e-06, 2.32011316e-06, 2.19657710e-06, 2.47194857e-06, 2.75105669e-06, 2.10990325e-06]

    # h_list = generate_h(h_mean_list)
    # print('h:',h_list)