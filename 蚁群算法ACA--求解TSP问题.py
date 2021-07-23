__time__ = '2021/7/14'
__author__ = 'ZhiYong Sun'
__doc__ = '常用的变异蚁群算法有：' \
          '1--精英蚂蚁系统----全局最优解决方案在每个迭代以及其他所有的蚂蚁的沉积信息素。' \
          '2--最大最小蚂蚁系统（MMAS）----添加的最大和最小的信息素量[ τmax，τmin ]，只有全局最佳或迭代最好的巡逻沉积的信息素。' \
          '所有的边缘都被初始化为τmax并且当接近停滞时重新初始化为τmax。' \
          '3--基于排序的蚂蚁系统（ASrank）----所有解决方案都根据其长度排名。然后为每个解决方案衡量信息素的沉积量，' \
          '最短路径相比较长路径的解沉积了更多的信息素。' \
          '4--连续正交蚁群（COAC）----COAC的信息素沉积机制能使蚂蚁协作而有效地寻解。利用正交设计方法，在可行域的蚂蚁' \
          '可以使用增大的全局搜索能力和精度，快速、高效地探索他们选择的区域。 ' \
          '正交设计方法和自适应半径调整方法也可推广到其他优化算法中，在解决实际问题施展更大的威力。'

"""优势--在图表动态变化的情况下解决相似问题时，他们相比模拟退火算法和遗传算法方法有优势；
蚁群算法可以连续运行并适应实时变化。这在网络路由和城市交通系统中是有利的。"""

"""蚁群算法思想--它旨在解决推销员问题，其目标是要找到一系列城市的最短遍历路线。总体算法相对简单，它基于一组蚂蚁，每只完成一次城市间的遍历。
在每个阶段，蚂蚁根据一些规则选择从一个城市移动到另一个：它必须访问每个城市一次;一个越远的城市被选中的机会越少（能见度更低）;
在两个城市边际的一边形成的信息素越浓烈，这边被选择的概率越大;如果路程短的话，已经完成旅程的蚂蚁会在所有走过的路径上沉积更多信息素，
每次迭代后，信息素轨迹挥发。"""


import numpy as np
import csv
from scipy import spatial

class MyACA_TSP:
    def __init__(self):
        self.dim = 48  # 城市数量
        self.pop_size = 10  # 蚂蚁数量
        self.max_iter = 20  # 迭代次数
        self.alpha = 1  # 信息素重要程度
        self.beta = 2  # 适应度的重要程度
        self.rho = 0.1  # 信息素挥发速度
        points_coordinate = self.read_data(r'C:\Users\Darius\Desktop\GA_Tsp_Text.csv')
        self.distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
        self.prob_matrix_distance = 1 / (self.distance_matrix + 1e-10 * np.eye(self.dim, self.dim))  # 避免除零错误

        self.Tau = np.ones((self.dim, self.dim))  # 信息素矩阵，每次迭代都会更新
        self.Table = np.zeros((self.pop_size, self.dim)).astype(int)  # 某一代每个蚂蚁的爬行路径
        self.y = None  # 某一代每个蚂蚁的爬行总距离
        self.generation_best_X, self.generation_best_Y = [], []  # 记录各代的最佳情况
        self.x_best_history, self.y_best_history = self.generation_best_X, self.generation_best_Y  # 历史原因，为了保持统一
        self.best_x, self.best_y = None, None

    def read_data(self, file_path):  # 读取文件
        distances = []
        with open(file_path, mode='r', encoding='utf-8') as fr:
            read = csv.reader(fr)
            for line in list(read):
                distances.append([int(line[1]), int(line[2])])
        return distances

    # 计算每个个体适应度函数值
    def func(self, routine):
        num_points, = routine.shape  # 取城市数量
        return sum([self.distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):  # 对每次迭代
            prob_matrix = (self.Tau ** self.alpha) * (self.prob_matrix_distance) ** self.beta  # 转移概率，无须归一化。
            for j in range(self.pop_size):  # 对每个蚂蚁
                self.Table[j, 0] = 0  # start point，其实可以随机，但没什么区别
                for k in range(self.dim - 1):  # 蚂蚁到达的每个节点
                    taboo_set = set(self.Table[j, :k + 1])  # 已经经过的点和当前点，不能再次经过
                    allow_list = list(set(range(self.dim)) - taboo_set)  # 在这些点中做选择
                    prob = prob_matrix[self.Table[j, k], allow_list]
                    prob = prob / prob.sum()  # 概率归一化
                    next_point = np.random.choice(allow_list, size=1, p=prob)[0]
                    self.Table[j, k + 1] = next_point

            # 计算距离
            y = np.array([self.func(i) for i in self.Table])

            # 顺便记录历史最好情况
            index_best = y.argmin()
            x_best, y_best = self.Table[index_best, :].copy(), y[index_best].copy()
            self.generation_best_X.append(x_best)
            self.generation_best_Y.append(y_best)

            # 计算需要新涂抹的信息素
            delta_tau = np.zeros((self.dim, self.dim))
            for j in range(self.pop_size):  # 每个蚂蚁
                for k in range(self.dim - 1):  # 每个节点
                    n1, n2 = self.Table[j, k], self.Table[j, k + 1]  # 蚂蚁从n1节点爬到n2节点
                    delta_tau[n1, n2] += 1 / y[j]  # 涂抹的信息素
                n1, n2 = self.Table[j, self.dim - 1], self.Table[j, 0]  # 蚂蚁从最后一个节点爬回到第一个节点
                delta_tau[n1, n2] += 1 / y[j]  # 涂抹信息素

            # 信息素飘散+信息素涂抹
            self.Tau = (1 - self.rho) * self.Tau + delta_tau

        best_generation = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[best_generation]
        self.best_y = self.generation_best_Y[best_generation]
        print(self.best_x, self.best_y)
        return self.best_x, self.best_y


if __name__ == "__main__":
    MyACA_TSP().run(200)
