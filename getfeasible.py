# 输入：决策变量精简后无线设备，各设备数据上传时延，及计算各设备本地执行时延的参数
# def getfeasibleres(edge_list, upload, recordD_i, recordg_i, recordf_i, E_min, C_i_l, C_i_e, E_i, H_get):
def getfeasibleres(edge_list, upload, recordD_i, recordg_i, recordf_i, E_min, C_i_l, C_i_e, E_i, T_):
    T = T_
    # 可行性分析判断lamuta是否满足条件
    # def analyselamta(m, E_min, C_i_l, C_i_e, E_i, H_get):
    #     tempres = []
    #     for i in range(len(m)):
    #         tempres.append((E_min[i] + (1 - m[i]) * C_i_l[i] + m[i] * C_i_e[i] - E_i[i]) / H_get[i])
    #     return max(tempres)

    # 可行性分析判断μ是否满足条件
    def analysemiu(m, upload, T):
        tempres = []
        for i in range(len(m)):
            tempres.append(m[i] * upload[i])
        return (T - sum(tempres) )

    tempact = []  # 记录该设备是边缘执行或本地执行的时延最小
    recordlantency = []  # 记录各设备时延最小决策的时延
    for i in range(len(edge_list)):
        if upload[i] < (recordD_i[i] * recordg_i[i] / recordf_i[i]):  # 该设备边缘执行时延更小
            tempact.append(1)
            recordlantency.append(upload[i])
        else:  # 该设备本地执行时延更小
            tempact.append(0)
            recordlantency.append(recordD_i[i] * recordg_i[i] / recordf_i[i])
    WDlist = edge_list[:]
    E_mintemp = E_min[:]
    C_ltemp = C_i_l[:]
    C_etemp = C_i_e[:]
    E_itemp = E_i[:]
    # H_gettemp = H_get[:]
    orderWD = []  # 记录按最小决策时延排序的各设备号
    orderact = []  # 记录按最小决策时延排序的各设备决策
    orderlantency = []  # 记录最小决策时延排序的各设备时延
    # 以下参数为分析lamta所用
    orderE_min = []  # 记录按最小决策时延排序后各设备最小能量
    orderC_l = []  # 记录按最小时延排序后各设备本地执行能耗
    orderC_e = []  # 记录按最小时延排序后各设备边缘执行能耗
    orderE_i = []  # 记录按最小时延排序后各设备剩余能量
    orderH_get = []  # 记录按最小时延排序后各设备采集能量
    num = len(edge_list)
    while num > 0:
        orderlantency.append(min(recordlantency))
        minindex = recordlantency.index(min(list(recordlantency)))
        orderWD.append(WDlist[minindex])
        orderact.append(tempact[minindex])
        # 判断当前设备是否满足lamta和μ的条件，若不满足将其决策改为0
        orderE_min.append(E_mintemp[minindex])
        orderC_l.append(C_ltemp[minindex])
        orderC_e.append(C_etemp[minindex])
        orderE_i.append(E_itemp[minindex])
        # orderH_get.append(H_gettemp[minindex])
        if tempact[minindex] == 1:
            # lamta = analyselamta(orderact, orderE_min, orderC_l, orderC_e, orderE_i, orderH_get)
            # eta = analysemiu(orderact, orderlantency, T)
            # # if lamta > eta or eta < 0:
            # if eta < 0:
            #     orderact[-1] = 0  # 不满足约束将当前分析的决策变为本地执行

            #当前设备 能量约束是否满足
            energy_limit = E_itemp[minindex] -  tempact[minindex] * C_etemp[minindex]
            if energy_limit < E_mintemp[minindex]:
                orderact[-1] = 0  # 不满足能量约束将当前分析的决策变为本地执行

            if analysemiu(orderact, orderlantency, T) < 0:
                orderact[-1] = 0

        # 去掉当前时延最小的设备，从剩下的设备中选择时延最小的设备
        del WDlist[minindex]
        del recordlantency[minindex]
        del tempact[minindex]
        del E_mintemp[minindex]
        del C_ltemp[minindex]
        del C_etemp[minindex]
        del E_itemp[minindex]
        # del H_gettemp[minindex]
        num -= 1
    num2 = len(orderWD)
    result = []
    while num2 > 0:
        a = min(orderWD)
        b = orderWD.index(a)
        result.append(orderact[b])
        del orderWD[b]
        del orderact[b]
        num2 -= 1
    return result


'''
edge_list=[0, 1, 2, 4, 5, 6, 7, 9]
upload=[0.7333795205427563, 1.5687668095424574, 1.575473989252263, 1.231936988330757, 1.5805804638349132,1.7376776251106258, 0.12549360941254248, 0.7215607625183167]
recordD_i=[44.8855806542653, 109.96938796230886, 106.56805410808647, 71.32905291437896, 99.38129120195913, 120.61345047945413, 8.558133962064574, 53.14692834562041]
recordg_i=[2.6292555820105212, 2.3603844051922027, 1.7177621630670763, 1.7332914220812827, 1.8733251972277023, 3.177289651783988, 3.330108183762311, 2.372683472448677]
recordf_i=[120.322781597372, 126.62556279289923, 120.6015006530047, 135.26985963173766, 157.9265911356294, 101.35338106101048, 95.82201616050006, 137.7688033480928]
E_min=[9.859568980124758, 8.459180405393196, 8.675966117988153, 9.237570538701148, 10.276263958329206, 8.303585701628265, 10.415904531289428, 9.967098615381545]
C_i_l=[1.7085802382741922e-21, 4.1619544407077786e-21, 2.6625360195606366e-21, 2.2622475371967027e-21, 4.643316978042748e-21, 3.936670194058366e-21, 2.616784940064252e-22, 2.393424579367492e-21]
C_i_e=[0.8680006900526974, 0.4590274888254641, 0.6277947692046567, 0.5987533877524651, 0.6274109248507638, 0.6297358761482441, 0.579537777261458, 0.6652717149195334]
E_i=[97.5985053540538, 100.12899323135095, 113.163672186868, 109.1647606289811, 108.3006632051836, 108.75318813611084, 100.1389104251639, 106.82238971734438]
H_get=[7.710973700403814e-05, 0.00026894163784855617, 0.0003007207843196407, 6.543334391892148e-05, 6.54044406459761e-05, 0.00013640341277900632, 7.698765436437608e-05, 0.00033601118114545475]
getfeasibleres(edge_list,upload,recordD_i,recordg_i,recordf_i,E_min,C_i_l,C_i_e,E_i,H_get)

'''
