__time__ = '2021/7/10'
__author__ = 'ZhiYong Sun'

from sko.GA import GA_TSP
import csv
from scipy import spatial
import numpy as np
np.set_printoptions(threshold=np.inf)


def read_data(file_path):  # 读取文件
    distances = []
    with open(file_path, mode='r', encoding='utf-8') as fr:
        read = csv.reader(fr)
        for line in list(read):
            distances.append([int(line[1]), int(line[2])])
    return distances


# 计算每个个体适应度函数值
def func(routine):
    num_points, = routine.shape  # 取城市数量
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


points_coordinate = read_data(r'C:\Users\Darius\Desktop\GA_Tsp_Text.csv')
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
ga_tsp = GA_TSP(func=func, n_dim=len(points_coordinate), size_pop=50, max_iter=200, prob_mut=0.001)
best_points, best_distance = ga_tsp.run()
print(best_points, best_distance)