import random
from sic_compute import *

class Gene:
    def get_location(self) -> tuple:
        #* 获得x坐标
        res_x = 0
        temp = 0.1
        for x in self.chromosome[0]:
            res_x += (x * temp)
        
        #* 获得y坐标
        res_y = 0
        temp = 0.1
        for x in self.chromosome[1]:
            res_y += (x * temp)
            temp / 10
        return (res_x, res_y)

    def judge(self) -> bool:
        res_x, res_y = self.get_location()
        #* 遍历每个设备，判断是否都在每个无线设备的传输范围内
        for device in self.devices:
            if (res_x - device.x)**2 + (res_y - device.y)**2 >= device.r:
                return False
        return True

    def __init__(self, devices:list, chromosome=None, location=None) -> None:
        self.devices = devices
        self.chromosome = chromosome
        if self.chromosome is not None:
            self.chromosome = chromosome
        elif location is not None:
            self.chromosome = [[],[]]
            x, y = (location[0] + location[1])/2, (location[2] + location[3])/2
            t = 0.1
            for i in range(15):
                temp_x = x // t
                temp_y = y // t
                self.chromosome[0].append(x)
                self.chromosome[1].append(y)

                x = x % t
                y = y % t
                t = t / 10

    def fitness(self) -> int:
        #* 下面为一些参数，这些需要自己设置
        N0 = 5          # 噪声功率
        alpha = 2       # 信号衰减系数
        beta = 0.2 
        location = self.get_location()
        server = Server(location[0], location[1])
        # return len(sic_h(devices_all=self.devices, server=server, alpha=alpha, N0=N0, beta=beta))
        return sic_h(devices_all=self.devices, server=server, N0=N0, beta=beta)

    #* 交叉
    def crossover(self, mother, crossover_rate:float) -> None:
        for k in range(2):
            child = [-1 for _ in range(len(self.chromosome[k]))]

            for i in range(len(self.chromosome[k])):
                temp = random.random()
                if temp < crossover_rate:
                    child[i] = self.chromosome[k][i]

            
            ptr1 = 0
            ptr2 = 0
            while(ptr1 < len(self.chromosome[k])):
                if self.chromosome[k][i] != -1:
                    ptr1 += 1
                    continue
                
                if mother.chromosome[k][ptr2] in self.chromosome[k]:
                    ptr2 += 1
                    continue
                
                self.chromosome[k][ptr1] = mother.chromosome[k][ptr2]
                ptr1 += 1
                ptr2 += 1


    #* 变异
    def mutation(self, mutation_rate:float) -> None:
        for k in range(2):
            for i in range(len(self.chromosome[k])):
                if random.random() < mutation_rate:
                    #* 随机生成一个基因，与当前基因进行交换达到变异的目的
                    temp = random.randint(0, 9)
                    for j in range(len(self.chromosome[k])):
                        if self.chromosome[k][j] == temp:
                            self.chromosome[k][i], self.chromosome[k][j] = self.chromosome[k][j], self.chromosome[k][i]
                            break


if __name__ == "__main__":
    n = 10
 
    devices, location = create_wireless_device(n, 1, 1, 0.2, 0.35, 0.2)

    figure, axes = plt.subplots()
    i = 1
    for device in devices:
        print("第%d个无线设备的位置是： x: %f,  y: %f,  r: %f" %(i, device.x, device.y, device.r))
        draw_circle = plt.Circle((device.x, device.y), device.r, color='b', fill=False)
        axes.add_artist(draw_circle)
        i += 1
    print(location)
    server = Server((location[0] + location[1])/2, (location[2] + location[3])/2)
    plt.scatter(server.x, server.y, s=10)
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))


    #* 设置种群内个体数目
    population_num = 200

    #* 生成种群
    population = []
    for i in range(population_num):
        temp = Gene(devices=devices, location=location)
        population.append(temp)
        print("第%d个个体生成完毕" % (i+1))

    for i in range(population_num):
        print("第 %d 个个体的适应度为: %.8f" % (i + 1, population[i].fitness()))
        

    #* 迭代次数times
    times = 100
    crossover_rate = 0.8
    mutation_rate = 0.3

    #* fitness_res存储了每次迭代后种群中适应值最大的个体，以便于画出适应度函数的进化曲线
    fitness_res = []

    for i in range(times):
        population.sort(key=lambda x:x.fitness(), reverse=True)
        temp_location = population[0].get_location()
        print("第 %d 次迭代得到的最大适应度为：%.8f。其中他的位置是x:%f. y:%f" \
            % (i+1, population[0].fitness(), temp_location[0], temp_location[1]))
        fitness_res.append(population[0].fitness())

        #* 计算种群适应值之和
        #* 考虑到存在负适应值的情况，所以先让每个个体的适应值加上最小的适应值（起到每个个体适应值大于0的作用）
        min_fitness = 9999
        res = 0
        for x in population:
            res += x.fitness() 
            min_fitness = min(min_fitness, x.fitness())
        res += population_num * min_fitness

        #* 通过轮盘赌确定两个parent的位置
        rate = 0
        father_position = 0
        for x in population:
            temp = random.random()
            rate += (x.fitness() + min_fitness) / res
            if temp < rate:
                break
            father_position += 1

        mother_position = 0
        rate = 0
        for x in population:
            temp = random.random()
            rate += (x.fitness() + min_fitness) / res
            if temp < rate:
                break
            mother_position += 1        

        #* 开始繁衍
        father_chromosome = population[father_position].chromosome.copy()
        mother_chromosome = population[mother_position].chromosome.copy()
        print(father_chromosome)
        print(type(father_chromosome))
        for _ in range(10):
            child = Gene(devices=devices, chromosome=father_chromosome)
            mother = Gene(devices=devices, chromosome=mother_chromosome)

            #* 交叉率0.8，变异率0.3
            child.crossover(mother, crossover_rate)
            child.mutation(mutation_rate)

            population.pop()
            population.insert(0, child)

    population.sort(key=lambda x:x.fitness(), reverse=True)
    print(population)


