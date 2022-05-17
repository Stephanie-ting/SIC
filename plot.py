import matplotlib.pyplot as plt
import numpy as np

def load_data(file_name):
    file = open(file_name)
    time_list = []
    for line in file.readlines():
        line = line.strip()
        line = float(line)
        time_list.append(line)
    # print(time_list)
    return time_list


if __name__ == '__main__':
    no_device = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    EAOOSIC_lantency_list = load_data('EAOOSIC_lantency.txt')
    EAOO_lantency_list = load_data('EAOO_lantency.txt')
    DROO_lantency_list = load_data('DROO_lantency.txt')
    local_lantency_list = load_data('local_lantency.txt')

    # 设置绘图风格
    plt.style.use('ggplot')

    # 绘制水平交错条形图
    bar_width = 0.2
    plt.bar(x=np.arange(len(no_device)), height=EAOOSIC_lantency_list, label='EAOO-SIC', color='steelblue', width=bar_width)
    plt.bar(x=np.arange(len(no_device)) + bar_width, height=EAOO_lantency_list, label='EAOO', color='powderblue', width=bar_width)
    plt.bar(x=np.arange(len(no_device)) + bar_width * 2, height=DROO_lantency_list, label='DROO',color='cornflowerblue', width=bar_width)
    plt.bar(x=np.arange(len(no_device)) + bar_width * 3, height=local_lantency_list, label='Fully Local', color='lightseagreen',width=bar_width)
    # 添加x轴刻度标签（向右偏移0.225）
    plt.xticks(np.arange(11) + 0.2, no_device)
    # 添加y轴标签
    plt.ylabel('Time Accomplishing Time')
    # 添加图形标题
    plt.title('Number of wireless terminals')
    # 添加图例
    plt.legend()
    # 显示图形
    plt.show()
