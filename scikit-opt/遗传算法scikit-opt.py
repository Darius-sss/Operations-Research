__time__ = '2021/7/10'
__author__ = 'ZhiYong Sun'
__doc__ = '---调用scikit-opt库中遗传算法如下---'


from sko.GA import GA
import math


def schaffer(p):  # sko.GA默认求最小
    """
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    """
    x = p
    return - (10 * math.sin(5 * x) + 7 * math.cos(4 * x))


ga = GA(func=schaffer, n_dim=1, size_pop=500, max_iter=500, prob_mut=0.001, lb=-20, ub=20, precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

