__time__ = '2021/7/12'
__author__ = 'ZhiYong Sun'

"""
简介
模拟退火来自冶金学的专有名词退火。退火是将材料加热后再经特定速率冷却，目的是增大晶粒的体积，并且减少晶格中的缺陷。材料中的原子原来会停留在使内能有局部最小值的位置，加热使能量变大，原子会离开原来位置，而随机在其他位置中移动。退火冷却时速度较慢，使得原子有较多可能可以找到内能比原先更低的位置。

模拟退火的原理也和金属退火的原理近似：我们将热力学的理论套用到统计学上，将搜寻空间内每一点想像成空气内的分子；分子的能量，就是它本身的动能；而搜寻空间内的每一点，也像空气分子一样带有“能量”，以表示该点对命题的合适程度。算法先以搜寻空间内一个任意点作起始：每一步先选择一个“邻居”，然后再计算从现有位置到达“邻居”的概率。

初始化
由一个产生函数从当前解产生一个位于解空间的新解，并定义一个足够大的数值作为初始温度。

迭代过程
迭代过程是模拟退火算法的核心步骤，分为新解的产生和接受新解两部分：

由一个产生函数从当前解产生一个位于解空间的新解；为便于后续的计算和接受，减少算法耗时，通常选择由当前新解经过简单地变换即可产生新解的方法，如对构成新解的全部或部分元素进行置换、互换等，注意到产生新解的变换方法决定了当前新解的邻域结构，因而对冷却进度表的选取有一定的影响。
计算与新解所对应的目标函数差。因为目标函数差仅由变换部分产生，所以目标函数差的计算最好按增量计算。事实表明，对大多数应用而言，这是计算目标函数差的最快方法。
判断新解是否被接受，判断的依据是一个接受准则，最常用的接受准则是Metropolis准则：若Δt′<0则接受S′作为新的当前解S，否则以概率exp（-Δt′/T）接受S′作为新的当前解S。
当新解被确定接受时，用新解代替当前解，这只需将当前解中对应于产生新解时的变换部分予以实现，同时修正目标函数值即可。此时，当前解实现了一次迭代。可在此基础上开始下一轮试验。而当新解被判定为舍弃时，则在原当前解的基础上继续下一轮试验。
模拟退火算法与初始值无关，算法求得的解与初始解状态S（是算法迭代的起点）无关；模拟退火算法具有渐近收敛性，已在理论上被证明是一种以概率1收敛于全局最优解的全局优化算法；模拟退火算法具有并行性。

停止准则
迭代过程的停止准则：温度T降至某最低值时，完成给定数量迭代中无法接受新解，停止迭代，接受当前寻找的最优解为最终解。

退火方案
在某个温度状态T下，当一定数量的迭代操作完成后，降低温度T，在新的温度状态下执行下一个批次的迭代操作。
"""

import numpy as np


class MySA:
    def __init__(self):
        self.max_iter = 500   # 最大迭代次数
        self.T_max = 100      # 最大温度，也是初始温度
        self.T_min = 1e-7     # 最小温度作为温度下界
        self.max_stay_counter = 150   # 如果max_stay_counter次内没有更新解，则结束

        x0 = [10]   # 初始解
        self.dim = len(x0)  # 初始解维度
        self.best_x = np.array(x0)
        self.best_y = self.calc_obj_value(self.best_x)   # 计算对应函数值
        self.T = self.T_max   # 当前温度
        self.iter_cycle = 0   # 记录迭代次数
        self.generation_best_X, self.generation_best_Y = [self.best_x], [self.best_y]  # 记录产生的新解
        # 记录历史解，用于计算max_stay_counter
        self.best_x_history, self.best_y_history = self.generation_best_X, self.generation_best_Y

    def calc_obj_value(self, x):  # 计算适应度函数
        best_y = 10 * np.sin(5 * x[0]) + 7 * np.cos(4 * x[0])
        return best_y

    def get_new_x(self, x):   # 产生新解
        u = np.random.uniform(-1, 1, size=self.dim)
        x_new = x + 20 * np.sign(u) * self.T * ((1 + 1.0 / self.T) ** np.abs(u) - 1.0)
        return x_new

    def cool_down(self):   # 退火
        self.T = self.T * 0.7

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=1e-30):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def run(self):
        x_current, y_current = self.best_x, self.best_y
        stay_counter = 0
        while True:
            for i in range(self.max_iter):
                x_new = self.get_new_x(x_current)
                y_new = self.calc_obj_value(x_new)

                # Metropolis准则
                df = y_new - y_current
                if df < 0 or np.exp(-df / self.T) > np.random.rand():
                    x_current, y_current = x_new, y_new
                    if y_new < self.best_y:
                        self.best_x, self.best_y = x_new, y_new

            self.iter_cycle += 1
            self.cool_down()
            self.generation_best_Y.append(self.best_y)
            self.generation_best_X.append(self.best_x)

            # if best_y stay for max_stay_counter times, stop iteration
            if self.isclose(self.best_y_history[-1], self.best_y_history[-2]):
                stay_counter += 1
            else:
                stay_counter = 0

            if self.T < self.T_min:
                print('Cooled to final temperature')
                break
            if stay_counter > self.max_stay_counter:
                print('Stay unchanged in the last {stay_counter} iterations'.format(stay_counter=stay_counter))
                break

        return self.best_x, self.best_y


if __name__ == "__main__":
    x, y = MySA().run()
    print(x, y)