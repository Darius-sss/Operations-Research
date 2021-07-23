__time__ = '2021/7/14'
__author__ = 'ZhiYong Sun'


from sko.ACA import ACA_TSP
import csv
from scipy import spatial
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import pandas as pd


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


aca = ACA_TSP(func=func, n_dim=48,
              size_pop=10, max_iter=20,
              distance_matrix=distance_matrix)

best_x, best_y = aca.run()
print(best_x, best_y)
# # %% Plot
# fig, ax = plt.subplots(1, 2)
# best_points_ = np.concatenate([best_x, [best_x[0]]])
# best_points_coordinate = points_coordinate[best_points_, :]
# ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
# pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
# plt.show()
