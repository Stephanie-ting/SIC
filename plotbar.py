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

def plot_time_n(EAOOSIC_filename,EAOO_filename, DROO_filename, local_filename, x_value):
    EAOOSIC = load_data(EAOOSIC_filename)
    EAOO = load_data(EAOO_filename)
    DROO = load_data(DROO_filename)
    local = load_data(local_filename)
    # 设置刻度范围
    plt.xlim(9, 32)
    plt.ylim(0, 45)
    # 设置刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.grid()
    plt.grid(zorder=0)

    n = x_value
    x_interval = n
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    total_width, m = 1.5, 4
    width = total_width / m

    for i in range(len(x_interval)):
        x_interval[i] = x_interval[i] * 1.0 - width
    plt.bar(x_interval, EAOOSIC, width=width, alpha=0.9, label='EAOOSIC', fc='steelblue', edgecolor='black', zorder=3)

    for i in range(len(x_interval)):
        x_interval[i] = x_interval[i] * 1.0 + width
    plt.bar(x_interval, EAOO, width=width, alpha=0.9, label='EAOO', fc='powderblue', edgecolor='black', zorder=3)

    for i in range(len(x_interval)):
        x_interval[i] = x_interval[i] * 1.0 + width
    plt.bar(x_interval, DROO, width=width, label='DROO',alpha=0.9, hatch='', fc='cornflowerblue', edgecolor='black', zorder=3)

    for i in range(len(x_interval)):
        x_interval[i] = x_interval[i] * 1.0 + width
    plt.bar(x_interval, local, width=width,alpha=0.9, label='Fully Local Computing', fc='lightseagreen', hatch='', edgecolor='black', zorder=3)

    xlabel = 'Number of Wireless Terminals'
    ylabel = 'Task Accomplishing Time (s)'
    x_major_locator = MultipleLocator(2)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(4)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlabel(xlabel, font2)
    plt.ylabel(ylabel, font2)
    # legend_font = {"family": "Times New Roman"}
    legend_font = {"family": "Times New Roman",
                   'weight': 'normal',
                   'size': 16,
                   }
    plt.legend(loc='upper left', prop=legend_font, edgecolor='black')
    plt.savefig('./TaskTime-N.eps', format='eps', dpi=1000)
    # plt.show()

    plt.show()
    plt.figure().tight_layout()

def plot_time_B(EAOOSIC_filename, EAOO_filename, DROO_filename, local_filename, x_value):
    EAOOSIC = load_data(EAOOSIC_filename)
    EAOO = load_data(EAOO_filename)
    DROO = load_data(DROO_filename)
    local = load_data(local_filename)
    # B = [_ for _ in range(10, 32, 2)]
    B = x_value
    x_interval = B
    # interval and number
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 10,
             }
    total_width, n = 1.39, 4
    width = total_width / n
    # 设置刻度范围
    plt.xlim(9, 31,2)
    plt.ylim(0, 7,1)
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
    x_major_locator = MultipleLocator(2)
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
    plt.legend(loc='lower left', prop=legend_font, edgecolor='black')
    plt.savefig('./B-TimeDelay.eps', format='eps', dpi=1000)
    plt.show()

def plot_time_D(EAOOSIC_filename, EAOO_filename, DROO_filename, local_filename, x_value):
    EAOOSIC = load_data(EAOOSIC_filename)
    EAOO = load_data(EAOO_filename)
    DROO = load_data(DROO_filename)
    local = load_data(local_filename)

    D = x_value
    x_interval = D
    # interval and number
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 10,
             }
    # 设置刻度范围
    plt.xlim(10, 215, 20)
    plt.ylim(0, 9, 1)
    total_width, n = 15, 4
    width = total_width / n
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
    plt.bar(x_interval, local, width=width, alpha=0.9, label='Fully Local Computing', fc='lightseagreen', hatch='',
            edgecolor='black', zorder=3)

    xlabel = 'Minimum Task Data Size (Mb)'
    ylabel = 'Task Accomplishing Time (s)'
    x_major_locator = MultipleLocator(20)
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
    legend_font = {"family": "Times New Roman",
                   'weight': 'normal',
                   'size': 12,
                   }
    # plt.legend(prop=legend_font)
    plt.legend(loc='lower right', prop=legend_font, edgecolor='black')
    plt.savefig('./MinData-TimeDelay.eps', format='eps', dpi=1000)
    plt.show()

def CPU_time():
    # cputime

    time_delay_EAOOSIC = load_data('./EAOOSIC_time.txt')
    time_delay_EAOO = load_data('./EAOO_time.txt')
    time_delay_DROO = load_data('./DROO_time.txt')
    time_delay_completely_local = load_data('./local_time.txt')

    x_list = [_ for _ in range(10, 32, 2)]

    plt.plot(x_list, time_delay_completely_local, label='Completely Local',
             ls='-', linewidth=2, color='lightseagreen',
             marker='^', markerfacecolor='skyblue', markersize=6)
    plt.plot(x_list, time_delay_DROO,
             ls='-', label='DROO', linewidth=2, color='cornflowerblue',
             marker='^', markerfacecolor='slateblue', markersize=6)
    plt.plot(x_list, time_delay_EAOO, label='EAOO',
             ls='-', linewidth=2, color='darkorchid',
             marker='^', markerfacecolor='violet', markersize=6)
    plt.plot(x_list, time_delay_EAOOSIC, label='EAOOSIC',
             ls='-', linewidth=2, color='steelblue',
             marker='^', markerfacecolor='steelblue', markersize=6)

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
    plt.ylim(0, 80)
    plt.grid()
    font_style = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 10,
                  }
    xlabel = 'Number of Wireless Terminals'
    ylabel = 'CPU Time'
    plt.xlabel(xlabel, font_style)
    plt.ylabel(ylabel, font_style)
    legend_font = {"family": "Times New Roman"}
    plt.legend(prop=legend_font)
    plt.savefig('./CPUtime.eps', format='eps', dpi=1000)
    plt.show()

def plot_time_R(EAOOSIC_filename, EAOO_filename, DROO_filename, local_filename, x_value):
    EAOOSIC = load_data(EAOOSIC_filename)
    EAOO = load_data(EAOO_filename)
    DROO = load_data(DROO_filename)
    local = load_data(local_filename)

    R = x_value
    x_interval = R
    # interval and number
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 10,
             }
    # 设置刻度范围
    plt.xlim(10, 215,20)
    plt.ylim(0, 18,1)
    total_width, n = 15, 4
    width = total_width / n
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
    plt.bar(x_interval, local, width=width, alpha=0.9, label='Fully Local Computing', fc='lightseagreen', hatch='',
            edgecolor='black', zorder=3)

    xlabel = 'Minimum local computing rate (MHz/s)'
    ylabel = 'Task Accomplishing Time (s)'
    x_major_locator = MultipleLocator(20)
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
    legend_font = {"family": "Times New Roman",
                   'weight': 'normal',
                   'size': 9,
                   }
    # plt.legend(prop=legend_font)
    plt.legend(loc='upper right', prop=legend_font, edgecolor='black')
    plt.savefig('./MinLocal-TimeDelay.eps', format='eps', dpi=1000)
    plt.show()



if __name__ == '__main__':

    n = [_ for _ in range(10, 32, 2)]
    plot_time_n('./EAOOSIC_lantency.txt','./EAOO_lantency.txt', './DROO_lantency.txt',  './local_lantency.txt', n)

    CPU_time()

    B = [_ for _ in range(10, 32, 2)]
    plot_time_B('./EAOOSIC_B_latency_list.txt','./EAOO_B_latency_list.txt', './DROO_B_latency_list.txt', './local_B_latency_list.txt', B)

    D_min = [_ for _ in range(20, 220, 20)]
    plot_time_D('./EAOOSIC_minData_latency_list.txt','./EAOO_minData_latency_list.txt', './DROO_minData_latency_list.txt', './local_minData_latency_list.txt', D_min)

    R = [_ for _ in range(20, 220, 20)]
    plot_time_R('./EAOOSIC_localrate_latency_list.txt','./EAOO_localrate_latency_list.txt','./DROO_localrate_latency_list.txt','./local_localrate_latency_list.txt',R)

