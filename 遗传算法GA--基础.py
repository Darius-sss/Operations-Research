__time__ = '2021/7/8'
__author__ = 'ZhiYong Sun'
__doc__ = '1-初始化种群' \
          '2-解码' \
          '3-计算适应度函数值' \
          '4-淘汰（适应度函数值为负数会对后面的轮盘赌选择造成影响，淘汰有助于加速搜索，下限设置过高会导致容易陷入局部最优）' \
          '5-轮盘赌选择' \
          '6-交叉' \
          '7-变异'


import math
import random
import matplotlib.pyplot as plt

from sko.GA import GA   # scikit-opt库， 和下面的GA不相关的

"""
求解的目标表达式为：
y = 10 * math.sin(5 * x) + 7 * math.cos(4 * x)
"""


# Genetic Algorithm
class MyGA:
    def __init__(self):

        self.pop_size = 500    # 种群数量
        self.upper_limit = 10  # 基因中允许出现的最大值
        self.y_min = 10   # 设置一个y的下界
        self.chromosome_length = 10  # 染色体长度
        self.iter = 500     # 迭代次数
        self.pc = 0.6   # 杂交概率
        self.pm = 0.01  # 变异概率
        self.results = []  # 存储每一代的最优解，N个二元组
        # pop = [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1] for i in range(pop_size)]
        self.pop = self.init_population()    # 随机生成初始种群

    # 随机生成初始种群
    def init_population(self):
        # 形如[[0,1,..0,1],[0,1,..0,1]...]
        pop = [[random.randint(0, 1) for i in range(self.chromosome_length)] for j in range(self.pop_size)]
        return pop

    # 计算2进制序列代表的数值
    def binary2decimal(self, binary):
        temp = 0
        for j in range(len(binary)):
            temp += binary[j] * 2 ** j
        temp = temp * self.upper_limit / (2 ** self.chromosome_length - 1)
        return temp

    # 解码并计算值, 二位列表 二进制数组 转换成 ->  实数
    def decode_chromosome(self):
        X = []
        for ele in self.pop:
            value = self.binary2decimal(ele)
            X.append(value)
        return X

    # 计算适应度函数值
    def calc_obj_value(self):
        obj_value = []
        X = self.decode_chromosome()
        for x in X:
            # 把缩放过后的那个数，带入我们要求的公式中
            # 种群中个体有几个，就有几个这种“缩放过后的数”
            obj_value.append(10 * math.sin(5 * x) + 7 * math.cos(4 * x))
        # 这里先返回带入公式计算后的数值列表，作为种群个体优劣的评价
        return obj_value

    # 淘汰, 负数在轮盘赌选择上会出现问题，如果y存在负数不淘汰的话需要进一步调整
    def calc_fit_value(self, obj_value):
        fit_value = []
        # 去掉小于0的值，更改c_min会改变淘汰的下限
        # 比如设成10可以加快收敛
        # 但是如果设置过大，有可能影响了全局最优的搜索
        c_min = self.y_min
        for value in obj_value:
            if value > c_min:
                temp = value
            else:
                temp = 0.
            fit_value.append(temp)
        # fit_value保存的是活下来的值
        return fit_value

    # 找出最优解和最优解的基因编码
    def find_best(self, fit_value):
        # 用来存最优基因编码
        best_individual = []
        # 先假设第一个基因的适应度最好
        best_fit = fit_value[0]
        for i in range(1, self.pop_size):
            if fit_value[i] > best_fit:
                best_fit = fit_value[i]
                best_individual = self.pop[i]
        # best_fit是值
        # best_individual是基因序列
        return best_individual, best_fit

    # 计算累计概率
    def cum_sum(self, fit_value):
        # 输入[1, 2, 3, 4, 5]，返回[1,3,6,10,15]，matlab的一个函数
        for i in range(1, self.pop_size):
            fit_value[i] += fit_value[i-1]

    # 轮赌法选择
    def selection(self, fit_value):
        p_fit_value = []
        # 适应度总和
        total_fit = sum(fit_value)
        # 归一化，使概率总和为1
        for i in range(len(fit_value)):
            p_fit_value.append(fit_value[i] / total_fit)   # 各自概率
        self.cum_sum(p_fit_value)   # 累计概率

        # 类似搞一个转盘吧下面这个的意思
        ms = sorted([random.random() for _ in range(self.pop_size)])   # 生成随机概率
        fit_ind = 0
        new_ind = 0
        newpop = [[] for _ in range(self.pop_size)]
        # 转轮盘选择法
        while new_ind < self.pop_size:
            # 如果这个概率大于随机出来的那个概率，就选这个
            if ms[new_ind] < p_fit_value[fit_ind]:
                newpop[new_ind] = self.pop[fit_ind]
                new_ind += 1
            else:
                fit_ind += 1
        # 这里注意一下，因为random.random()不会大于1，所以保证这里的newpop规格会和以前的一样
        # 而且这个pop里面会有不少重复的个体，保证种群数量一样
        self.pop = newpop[:]

    # 杂交
    def crossover(self):
        # 一定概率杂交，主要是杂交种群种相邻的两个个体
        for i in range(self.pop_size - 1):
            # 随机看看达到杂交概率没
            if random.random() < self.pc:
                # 随机选取杂交点，然后交换数组
                cpoint = random.randint(0, self.chromosome_length)
                self.pop[i], self.pop[i + 1] = self.pop[i][0:cpoint] + self.pop[i + 1][cpoint:len(self.pop[i])], \
                                               self.pop[i + 1][0:cpoint] + self.pop[i][cpoint:len(self.pop[i])]

    # 基因突变
    def mutation(self):
        # 每条染色体随便选一个杂交
        for i in range(self.pop_size):
            if random.random() < self.pm:
                mpoint = random.randint(0, self.chromosome_length - 1)
                self.pop[i][mpoint] = (self.pop[i][mpoint]+1) % 2   # 1 -> 0,  0 -> 1

    # 绘制求解函数的图，大致确定下限用于淘汰
    def plot_obj_func(self):
        """y = 10 * math.sin(5 * x) + 7 * math.cos(4 * x)"""
        X1 = [i / float(10) for i in range(0, 100, 1)]
        Y1 = [10 * math.sin(5 * x) + 7 * math.cos(4 * x) for x in X1]
        plt.plot(X1, Y1)
        plt.show()

    def run(self):
        print('y = 10 * math.sin(5 * x) + 7 * math.cos(4 * x)')
        self.plot_obj_func()

        for i in range(self.iter):
            obj_value = self.calc_obj_value()  # 计算适应度函数值
            fit_value = self.calc_fit_value(obj_value)  # 根据适应度函数值进行优胜劣汰，低于y_min的置为0
            best_individual, best_fit = self.find_best(fit_value)  # # 找出最优解的基因编码和最优值
            # 下面这句就是存放每次迭代的最优x值是最佳y值
            self.results.append([self.binary2decimal(best_individual), best_fit])

            self.selection(fit_value)  # 选择
            self. crossover()  # 染色体交叉（最优个体之间进行0、1互换）
            self.mutation()  # 染色体变异（其实就是随机进行0、1取反）

        print("x = %f, y = %f" % (self.results[-1][0], self.results[-1][1]))


if __name__ == '__main__':
    MyGA().run()




