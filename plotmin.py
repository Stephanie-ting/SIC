# coding=utf8
import matplotlib.pyplot as plt
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
    total_width, n = 7, 4
    width = total_width / n

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
    x_major_locator = MultipleLocator(10)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(2.0)
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
                   'size': 16,
                   }
    # plt.legend(prop=legend_font)
    plt.legend(loc='upper left', prop=legend_font, edgecolor='black')
    # plt.savefig('./B-TimeDelay.eps', format='eps', dpi=1000)
    plt.show()

if __name__ == '__main__':
    D_min = [_ for _ in range(50, 160, 10)]
    plot_time_D('./EAOOSIC_minData_latency_list.txt','./EAOO_minData_latency_list.txt', './DROO_minData_latency_list.txt', './local_minData_latency_list.txt', D_min)
