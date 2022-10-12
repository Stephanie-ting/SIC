import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import scipy.io as sio

def load_data(file_name):
    file = open(file_name)
    time_list = []
    for line in file.readlines():
        line = line.strip()
        line = float(line)
        time_list.append(line)
    # print(time_list)
    return time_list

def plot_cost_EAOOSIC(memory, color):
    import matplotlib.pyplot as plt
    for i in range(len(memory)):
        #EAOOSIC_cost = load_data('cost_EAOOSIC_' + str(memory[i]) + '.txt')
        EAOOSIC_cost = sio.loadmat('./memory/latency_EAOOSIC_%d' % memory[i])['memory_latency_list']
        EAOOSIC_cost = EAOOSIC_cost[0, :].tolist()
        plt.plot(np.arange(len(EAOOSIC_cost)) * 10, EAOOSIC_cost, color=color[i], label= 'Memory size = '+ str(memory[i]))
    plt.ylabel('EAOOSIC Training Loss')
    plt.xlabel('Time Frames')
    x_major_locator = MultipleLocator(250)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.1)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(0, 3000)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0.1, 1)
    plt.grid()
    legend_font = {"family": "Times New Roman",
                   'weight': 'normal',
                   'size': 16,
                   }
    plt.legend(loc='upper right', prop=legend_font, edgecolor='black')
    plt.show()

if __name__ == '__main__':
    memory = [64,128, 256, 512, 1024]
    color = ['lightseagreen', 'cornflowerblue', 'darkorchid', 'midnightblue', 'cadetblue']
    plot_cost_EAOOSIC(memory, color)