import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

def load_data(file_name):
    file = open(file_name)
    time_list = []
    for line in file.readlines():
        line = line.strip()
        line = float(line)
        time_list.append(line)
    # print(time_list)
    return time_list

def plot_time_noisep(EAOOSIC_filename, EAOO_filename, DROO_filename, local_filename, x_value):
    EAOOSIC = load_data(EAOOSIC_filename)
    EAOO = load_data(EAOO_filename)
    DROO = load_data(DROO_filename)
    local = load_data(local_filename)
    # B = [_ for _ in range(10, 32, 2)]
    noise = x_value
    # x_interval = noise
    x_interval = [10,20,30,40,50,60,70,80,90,100]
    # interval and number
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 10,
             }
    total_width, n = 1.39, 4
    width = total_width / n
    # 设置刻度范围
    plt.xlim(9, 31,2)
    plt.ylim(0, 9,1)
    # 设置刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.grid()
    plt.grid(zorder=0)

    for i in range(len(x_interval)):
        x_interval[i] = x_interval[i] * 1.0 - width
    plt.bar(x_interval, EAOOSIC, width=width, alpha=0.9, label='EAOOSIC', fc='steelblue', edgecolor='black', zorder=3)

    for i in range(len(x_interval)):
        x_interval[i] = x_interval[i] * 1.0 + width
    plt.bar(x_interval, EAOO, width=width, alpha=0.9, label='EAOO', fc='powderblue', edgecolor='black', zorder=3)

    for i in range(len(x_interval)):
        x_interval[i] = x_interval[i] * 1.0 + width
    plt.bar(x_interval, DROO, width=width, label='DROO', alpha=0.9, hatch='', fc='cornflowerblue', edgecolor='black',
            zorder=3)

    for i in range(len(x_interval)):
        x_interval[i] = x_interval[i] * 1.0 + width
    plt.bar(x_interval, local, width=width, alpha=0.9, label='Fully Local Computing', fc='lightseagreen', hatch='',edgecolor='black', zorder=3)

    xlabel = 'Channel Bandwidth'
    ylabel = 'Task Accomplishing Time'
    x_major_locator = MultipleLocator(10)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(1)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlabel(xlabel, font2)
    plt.ylabel(ylabel, font2)
    legend_font = {"family": "Times New Roman", 'weight': 'normal'}
    # plt.legend(prop=legend_font)
    plt.legend(loc='upper right', prop=legend_font, edgecolor='black')
    plt.savefig('./noisep-TimeDelay.eps', format='eps', dpi=1000)
    plt.show()


if __name__ == '__main__':

    noisep = [1e-9,2e-9, 3e-9,4e-9,5e-9,6e-9,7e-9,8e-9,9e-9,10e-9]
    plot_time_noisep('./noisep/EAOOSIC_np_latency_list.txt','./noisep/EAOO_np_latency_list.txt', './noisep/DROO_np_latency_list.txt', './noisep/local_np_latency_list.txt', noisep)
