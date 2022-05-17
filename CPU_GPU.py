# coding=utf8
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


def CPU_GPU_time():


    time_delay_EAOOSIC_CPU = load_data('./EAOOSIC_time_CPU.txt')
    time_delay_EAOOSIC_GPU = load_data('./EAOOSIC_time_GPU.txt')

    x_list = [_ for _ in range(10, 32, 2)]


    plt.plot(x_list, time_delay_EAOOSIC_CPU, label='SIC-CPU',
             ls='-', linewidth=2, color='steelblue',
             marker='^', markerfacecolor='steelblue', markersize=6)
    plt.plot(x_list, time_delay_EAOOSIC_GPU, label='SIC-GPU',
             ls='-', linewidth=2, color='cornflowerblue',
             marker='^', markerfacecolor='slateblue', markersize=6)

    x_major_locator = MultipleLocator(2)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(5.0)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(10, 31)

    plt.grid()
    font_style = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 10,
                  }
    xlabel = 'Number of Wireless Terminals'
    ylabel = 'CPU-GPU Time (s)'
    plt.xlabel(xlabel, font_style)
    plt.ylabel(ylabel, font_style)
    legend_font = {"family": "Times New Roman"}
    plt.legend(prop=legend_font)
    #plt.savefig('./CPUtime.eps', format='eps', dpi=1000)
    plt.show()




if __name__ == '__main__':



    CPU_GPU_time()

