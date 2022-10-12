# coding=utf8
# 输入：决策变量精简后无线设备，各设备数据上传时延，及计算各设备本地执行时延的参数

from sic_compute import *
def getfeasibleres(edge_list, upload, recordD_i, recordg_i, recordf_i, E_min, C_i_l, C_i_e, E_i,wireless_devices_,server_,N_0_,beta_,T_):
    WDlist = edge_list[:]
    E_mintemp = E_min[:]
    C_ltemp = C_i_l[:]
    C_etemp = C_i_e[:]
    E_itemp = E_i[:]

    wireless_devices = wireless_devices_
    server = server_
    # alpha = alpha_
    N_0 = N_0_
    beta = beta_
    T = T_
    A_d = 4.11  # 天线增益
    d_e = 2.8  # 路径损失指数
    f_c = 915e6  # 载波频率
    #n = n_

    #wireless_devices, location = create_wireless_device(n, 1, 1, 0.2, 0.35, 0.1)
    #server = Server((location[0] + location[1]) / 2, (location[2] + location[3]) / 2)

    edge_devices = []
    recordpower = []  # 记录各设备的p*h值
    #将edge_list里的设备，转换成自定义的wireless_devices结构
    for index in edge_list:
        edge_devices.append(wireless_devices[index])
    for device in edge_devices:
        distance_to = ((device.x - server.x) ** 2 + (device.y - server.y) ** 2) ** 0.5
        # 计算每个h
        t = 3e8 / (4 * 3.14 * f_c * distance_to)
        h_i = A_d * (t ** d_e)

        # 将p * h 的形式存入到数组中
        recordpower.append(device.power * h_i)

    tempact = []  # 记录该设备是边缘执行或本地执行的时延最小
    #recordlantency = []  # 记录各设备时延最小决策的时延

    for i in range(len(edge_list)):
        if upload[i] < (recordD_i[i] * recordg_i[i] / recordf_i[i]):  # 该设备边缘执行时延更小
            tempact.append(1)
            #recordlantency.append(upload[i])
            #recordpower.append()
        else:  # 该设备本地执行时延更小
            tempact.append(0)
            #recordlantency.append(recordD_i[i] * recordg_i[i] / recordf_i[i])

    orderWD = []  # 记录按最小决策时延排序的各设备号
    orderact = []  # 记录按最小决策时延排序的各设备决策
    #orderlantency = []  # 记录最小决策时延排序的各设备时延
    orderpower = []   #记录各设备排序后的p*h值

    current_up = []  # 存的下标
    WDorigin = edge_list[:]

    num = len(edge_list)
    while num > 0:
        #orderlantency.append(min(recordlantency))
        orderpower.append(max(recordpower))
        #minindex = recordlantency.index(min(recordlantency))   #选取的这个时延最小的设备下标
        minindex = recordpower.index(max(recordpower))  #选取的这个p*h大的设备下标
        orderWD.append(WDlist[minindex])
        orderact.append(tempact[minindex])


        #将orderact中边缘和本地的设备分开

        if tempact[minindex] == 1:
                current_up.append(WDlist[minindex])



        # *对上传的设备进行分组
        def split_group_lantency(current_up):

            #current_local = []  # 存的下标
            current_single = []  # 存的下标
            current_par = []  # 存的下标
            maxPar_lantency = []  # 存放每一组中时延最大的设备的所需时延  是时延


            # 从wireless_devices中取current_up中对应下标的设备
            up_devices = []
            for index in current_up:
                up_devices.append(wireless_devices[index])
            # split_list = sic(devices_all=up_devices, server=server, alpha=alpha, N0=N_0, beta=beta)
            split_list = sic_h(devices_all=up_devices, server=server, N0=N_0, beta=beta)

            # print("当前m的分组情况是：", split_list)
            # 分组情况是： [[9, 7, 6, 1, 8, 0], [4, 3, 2], [5]]
            # current_single[] and current_par

            for list in split_list:
                if len(list) != 1:
                    current_par.append(list)  # [[9, 7, 6, 1, 8, 0], [4, 3, 2]]
                else:
                    current_single.append(list)


            for par in current_par:  # par = [9, 7, 6, 1, 8, 0]
                par_lantency = []  # 存放每组设备上传时延
                for device in par:
                    par_lantency.append(upload[WDorigin.index(device)])
                maxPar_lantency.append(max(par_lantency))

            # 不能并行上传的设备的总时延   current_single = [[5]]
            single_lantency = 0
            for single_list in current_single:  # [5]
                for device in single_list:
                    single_lantency += upload[WDorigin.index(device)]
            return single_lantency, maxPar_lantency



        # * 约束条件判断
        if tempact[minindex] == 1:      #对当前新加的设备进行分析

            energy_limit = E_itemp[minindex] - ((1 - tempact[minindex]) * C_ltemp[minindex] + tempact[minindex] * C_etemp[minindex])

            #能量约束是否满足
            if energy_limit < E_mintemp[minindex]:
                orderact[-1] = 0  # 不满足能量约束将当前分析的决策变为本地执行

            #先对新加进来的设备进行sic判断，分组 之后 进行时延约束判断
            single_lantency, maxPar_lantency = split_group_lantency(current_up)

            time_limit = sum(maxPar_lantency) + single_lantency
            if time_limit > T:
                orderact[-1] = 0  # 不满足时延约束将当前分析的决策变为本地执行

        # 去掉当前时延最小的设备，从剩下的设备中选择时延最小的设备
        del WDlist[minindex]
        #del recordlantency[minindex]
        del recordpower[minindex]
        del tempact[minindex]
        del E_mintemp[minindex]
        del C_ltemp[minindex]
        del C_etemp[minindex]
        del E_itemp[minindex]
        num -= 1

    num2 = len(orderWD)
    #print("num2:",num2)
    result = []
    while num2 > 0:
        a = min(orderWD)
        b = orderWD.index(a)
        result.append(orderact[b])
        del orderWD[b]
        del orderact[b]
        num2 -= 1
    #print("保底可行解：", result)
    return result



if __name__ == "__main__":
    N_0 = 1e-10  # 接收端噪声功率
    T = 2
    alpha = 2  # 信号衰减系数
    beta = 0.5  # 阈值

    wireless_devices, location = create_wireless_device(10, 1, 1, 0.2, 0.35, 0.1)
    server = Server((location[0] + location[1]) / 2, (location[2] + location[3]) / 2)

    edge_list=[0, 1, 2, 4, 5, 6, 7, 9]
    upload=[0.7333795205427563, 1.5687668095424574, 1.575473989252263, 1.231936988330757, 1.5805804638349132,1.7376776251106258, 0.12549360941254248, 0.7215607625183167]
    recordD_i=[44.8855806542653, 109.96938796230886, 106.56805410808647, 71.32905291437896, 99.38129120195913, 120.61345047945413, 8.558133962064574, 53.14692834562041]
    recordg_i=[2.6292555820105212, 2.3603844051922027, 1.7177621630670763, 1.7332914220812827, 1.8733251972277023, 3.177289651783988, 3.330108183762311, 2.372683472448677]
    recordf_i=[120.322781597372, 126.62556279289923, 120.6015006530047, 135.26985963173766, 157.9265911356294, 101.35338106101048, 95.82201616050006, 137.7688033480928]
    E_min=[9.859568980124758, 8.459180405393196, 8.675966117988153, 9.237570538701148, 10.276263958329206, 8.303585701628265, 10.415904531289428, 9.967098615381545]
    C_i_l=[1.7085802382741922e-21, 4.1619544407077786e-21, 2.6625360195606366e-21, 2.2622475371967027e-21, 4.643316978042748e-21, 3.936670194058366e-21, 2.616784940064252e-22, 2.393424579367492e-21]
    C_i_e=[0.8680006900526974, 0.4590274888254641, 0.6277947692046567, 0.5987533877524651, 0.6274109248507638, 0.6297358761482441, 0.579537777261458, 0.6652717149195334]
    E_i=[97.5985053540538, 100.12899323135095, 113.163672186868, 109.1647606289811, 108.3006632051836, 108.75318813611084, 100.1389104251639, 106.82238971734438]
    #H_get=[7.710973700403814e-05, 0.00026894163784855617, 0.0003007207843196407, 6.543334391892148e-05, 6.54044406459761e-05, 0.00013640341277900632, 7.698765436437608e-05, 0.00033601118114545475]
    #getfeasibleres(edge_list,upload,recordD_i,recordg_i,recordf_i,E_min,C_i_l,C_i_e,E_i)


    #edge_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #upload=[0.375275825385459, 0.2660855091523818, 0.36434958562180214, 0.3446591807606277, 0.38557798974791424,
             #0.33335937083787787, 0.3056901241871115, 0.2505801030902589, 0.4008510353214122, 0.2568270967718143]
    #recordD_i=[136.31821076568764, 112.35937361679336, 149.00296901525283, 120.23686353931029, 134.82053788434973,
                #128.7516265907887, 129.5122413218225, 101.26443881119513, 118.16564894670057, 110.35029162636751]
    #recordg_i=[2.775592989235088, 2.414087204048344, 2.935252569467939, 2.015185607416563, 2.120752541598743,
                #2.78223765348545, 2.078906055177168, 2.624041906146043, 2.3940152219241773, 2.8941841060035984]
  #  recordf_i=[55.472303897544464, 80.50119107760786, 133.04593592242838, 103.54857951661906, 87.76695946022068,
                #92.45146834110082, 98.49912536569545, 116.82342013761969, 53.968424622878395, 77.72025909164302]
  #  E_min_rec=[12.896814984837519, 18.103328160047486, 10.353558280439653, 11.393929515733486, 19.250045584851936,
         #       12.999149752843435, 12.885287131856398, 19.275646668435435, 10.904973265277572, 10.599409849771149]
   # C_local_rec=[1.1642924094272698e-21, 1.7577895398166897e-21, 7.741829900430288e-21, 2.5980109835942836e-21,
   #               2.2024606508577008e-21, 3.0617841786782323e-21, 2.612224099440852e-21, 3.6264989816863655e-21,
      #            8.2394388252537885e-22, 1.9291594239760668e-21]
  #  C_up_rec=[0.19480028043340822, 0.15022493089556663, 0.20641018286225846, 0.20059428190308726, 0.20284721639531095,
    #           0.19960023739104113, 0.1725132056047888, 0.12937953992702475, 0.2338321766674948, 0.13555627645613935]
   # E_i_record=[569.7425677170725, 545.2804498594171, 590.6390475086133, 526.1009173433171, 596.8188657596088,
       #          528.9402723688594, 599.1490380474088, 526.9528516800722, 502.03720685571693, 544.4615646942332]

    result = getfeasibleres(edge_list, upload, recordD_i, recordg_i, recordf_i, E_min,C_i_l,C_i_e,E_i,wireless_devices,server,alpha,N_0,beta,T)
    #result = getfeasibleres(edge_list, upload, recordD_i, recordg_i, recordf_i, E_min_rec, C_local_rec, C_up_rec, E_i_record,
                          #  wireless_devices, server, alpha, N_0, beta, T)
    print("保底可行解：",result)
