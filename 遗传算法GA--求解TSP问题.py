__time__ = '2021/7/8'
__author__ = 'ZhiYong Sun'

__doc__ = '1-初始化种群' \
          '2-解码' \
          '3-计算适应度函数值' \
          '4-轮盘赌选择' \
          '5-交叉' \
          '6-变异'


import math
import random
import csv


class MyGATsp:
    def __init__(self):
        self.pop_size = 500  # 种群数量
        self.iter = 500  # 迭代次数
        self.pc = 0.6  # 杂交概率
        self.pm = 0.01  # 变异概率
        self.results = []  # 存储每一代的最优解，N个二元组
        self.distances = self.read_data(file_path=r'C:\Users\Darius\Desktop\GA_Tsp_Text.csv')
        self.chromosome_length = len(self.distances)  # 染色体长度
        # pop = [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1] for i in range(pop_size)]
        self.pop = self.init_population()  # 随机生成初始种群

    @classmethod
    def read_data(cls, file_path):    # 读取文件 组合成  {'1': ('6734', '1453'), '2': ('2233', '10'), ...} 格式
        distances = {}
        with open(file_path, mode='r', encoding='utf-8') as fr:
            read = csv.reader(fr)
            for line in list(read):
                distances[int(line[0])] = (int(line[1]), int(line[2]))
        return distances

    def init_population(self):
        """初始化种群"""
        lives = []
        for i in range(self.pop_size):
            gene = [x for x in range(1, self.chromosome_length+1)]
            random.shuffle(gene)
            lives.append(gene)  # 随机洗牌
        return lives

    # 计算每个个体适应度函数值, 返回 1/总距离 * pop_size
    def calc_obj_value(self):
        obj_values = []
        for i in range(self.pop_size):
            _sum = 0
            for ind in range(self.chromosome_length-1):
                curr, next_ = self.pop[i][ind], self.pop[i][ind+1]
                _sum += math.sqrt((self.distances[curr][1]-self.distances[next_][1])**2 + (self.distances[curr][0]-self.distances[next_][0]) ** 2)
            _sum += math.sqrt((self.distances[self.pop[i][0]][1]-self.distances[self.pop[i][-1]][1])**2 + \
                              (self.distances[self.pop[i][0]][0]-self.distances[self.pop[i][-1]][0]) ** 2)  # 连回起点
            obj_values.append(1/_sum)
        return obj_values

    # 找出最优解和最优解的基因编码
    def find_best(self, obj_values):
        # 先假设第一个基因的适应度最好
        best_individual = self.pop[0]
        best_fit = obj_values[0]
        for i in range(1, self.pop_size):
            if obj_values[i] > best_fit:
                best_fit = obj_values[i]
                best_individual = self.pop[i]
        # best_fit是值
        # best_individual是基因序列
        return best_individual, best_fit

    # 计算累计概率
    def cum_sum(self, obj_values):
        # 输入[1, 2, 3, 4, 5]，返回[1,3,6,10,15]，matlab的一个函数
        for i in range(1, self.pop_size):
            obj_values[i] += obj_values[i - 1]

    # 轮赌法选择
    def selection(self, obj_values):
        p_obj_values = []
        # 适应度总和
        total_fit = sum(obj_values)
        # 归一化，使概率总和为1
        for i in range(len(obj_values)):
            p_obj_values.append(obj_values[i] / total_fit)  # 各自概率
        self.cum_sum(p_obj_values)  # 累计概率

        # 类似搞一个转盘吧下面这个的意思
        ms = sorted([random.random() for _ in range(self.pop_size)])  # 生成随机概率
        fit_ind = 0
        new_ind = 0
        newpop = [[] for _ in range(self.pop_size)]
        # 转轮盘选择法
        while new_ind < self.pop_size:
            # 如果这个概率大于随机出来的那个概率，就选这个
            if ms[new_ind] < p_obj_values[fit_ind]:
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
        for i in range(self.pop_size-1):
            new_gene = []
            # 随机看看达到杂交概率没
            if random.random() < self.pc:
                index1 = random.randint(0, self.chromosome_length - 2)  # 随机生成突变起始位置 #
                index2 = random.randint(index1+1, self.chromosome_length - 1)  # 随机生成突变终止位置 #
                temp_gene = self.pop[i + 1][index1:index2]  # 交叉的基因片段
                p1len = 0
                for g in self.pop[i]:
                    if p1len == index1:
                        new_gene.extend(temp_gene)  # 插入基因片段
                    if g not in set(temp_gene):
                        new_gene.append(g)
                    p1len += 1
                self.pop[i] = new_gene[:]

    def mutation(self):
        """突变"""
        # 每条染色体随便选一个杂交
        for i in range(self.pop_size):
            if random.random() < self.pm:
                index1 = random.randint(0, self.chromosome_length - 1)
                index2 = random.randint(0, self.chromosome_length - 1)
                # 随机选择两个位置的基因交换--变异 #
                self.pop[i][index1], self.pop[i][index2] = self.pop[i][index2], self.pop[i][index1]

    def run(self):

        for i in range(self.iter):
            obj_values = self.calc_obj_value()  # 计算适应度函数值
            best_individual, best_fit = self.find_best(obj_values)  # # 找出最优解的基因编码和最优值
            # 下面这句就是存放每次迭代的最优x值是最佳y值
            self.results.append([best_individual, best_fit])
            print('迭代次数：', i, '当前最优解：', self.results[-1][0], 1/self.results[-1][1])
            self.selection(obj_values)  # 选择
            self. crossover()  # 染色体交叉（最优个体之间进行片段互换）
            self.mutation()  # 染色体变异（其实就是随机进行内部交换）

        print(self.results[-1][0], self.results[-1][1])


if __name__ == "__main__":
    MyGATsp().run()
