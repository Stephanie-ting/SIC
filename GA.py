# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:38:32 2021

@author: zc
"""

import numpy as np


class GA:
    def __init__(self,
                 n,     # 设备总数
                 s, x   # 种群
                 ):
        self.n = n
        self.s = list(s)
        if x:
            self.s.extend(x)
        self.pop_size = len(s)

    def __mutation(self, solution):
        if self.mutation_pos:
            pos = np.random.choice(a=self.mutation_pos)
            solution[pos] = 1 - solution[pos]            # 变异
        return solution

    def __cross(self, a, b):
        pos = np.random.randint(self.n)                  # 获取交叉点
        a[pos:] = b[pos:]
        return a

    def crossAndMutation(self, mutation_pos):
        new_pop = []
        self.mutation_pos = mutation_pos
        for f in self.s:
            m = self.s[np.random.randint(self.pop_size)] # 获取交叉的另一个样本
            child = self.__cross(f, m)                   # 交叉互换
            child = self.__mutation(child)               # 变异
            new_pop.append(child)
        self.s = new_pop


if __name__ == "__main__":
    s = [[0, 1, 0], [1, 0, 0]]
    x = [[0, 0, 0], [1, 1, 1]]
    g = GA(3, s, x)
    g.crossAndMutation(range(3))
    print(g.s)
