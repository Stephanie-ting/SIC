import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

def plot_ratio_EAOOSIC(memory, color):
    import matplotlib.pyplot as plt
    rolling_intv = 70
    for i in range(len(memory)):
        #EAOOSIC_cost = load_data('cost_EAOOSIC_' + str(memory[i]) + '.txt')
        EAOOSIC_ratio = sio.loadmat('./ratio/batch_EAOOSIC_%d' % memory[i])['ratio_batch_list']
        EAOOSIC_ratio = EAOOSIC_ratio[0, :].tolist()

        ratio_array = np.asarray(EAOOSIC_ratio)
        df = pd.DataFrame(ratio_array)


        plt.plot(np.arange(len(ratio_array)) + 1, df.rolling(rolling_intv, min_periods=1).mean(), color=color[i], label= 'Memory size = '+ str(memory[i]))
        # plt.fill_between(np.arange(len(ratio_array)) + 1, df.rolling(rolling_intv, min_periods=1).min()[0],df.rolling(rolling_intv, min_periods=1).max()[0], color='b', alpha=0.2)
        # plt.plot(np.arange(len(EAOOSIC_cost)) * 10, EAOOSIC_cost, color=color[i], label= 'Memory size = '+ str(memory[i]))
    plt.ylabel('SIC Approximation Ratio')
    plt.xlabel('Time Frames')
    x_major_locator = MultipleLocator(300)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.1)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(0, 300)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0, 1)

    legend_font = {"family": "Times New Roman",
                   'weight': 'normal',
                   'size': 12,
                   }
    plt.legend(loc='lower right', prop=legend_font, edgecolor='black')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    memory = [64,128, 256, 512]
    color = ['lightseagreen', 'cornflowerblue', 'darkorchid', 'midnightblue']
    plot_ratio_EAOOSIC(memory, color)