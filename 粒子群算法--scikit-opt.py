__time__ = '2021/7/10'
__author__ = 'ZhiYong Sun'
__doc__ = '---调用scikit-opt库中粒子群算法如下---'

from sko.PSO import PSO
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return 10 * np.sin(5 * x) + 7 * np.cos(4 * x)


pso = PSO(func=func, n_dim=1, pop=40, max_iter=150, lb=[-100], ub=[100], w=0.8, c1=0.5, c2=0.5)
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
plt.plot(pso.gbest_y_hist)
plt.show()