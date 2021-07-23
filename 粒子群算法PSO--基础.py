__time__ = '2021/7/10'
__author__ = 'ZhiYong Sun'
__doc__ = '名称--粒子群算法：粒子群算法的基本思想是模拟鸟群随机搜寻食物的捕食行为，鸟群通过自身经验和种群之间的交流调整自己的搜寻路径，' \
          '从而找到食物最多的地点' \
          'func--函数' \
          'dim--函数中参数维度' \
          'pop_size--种群规模：粒子群算法的最大特点就是速度快，因此初始种群取50-1000都是可以的，虽然初始种群越大收敛性会更好，不过太大了也会影响速度' \
          'max_iter--最大迭代次数：一般取100~4000，太少解不稳定，太多浪费时间。对于复杂问题，进化代数可以相应地提高' \
          'w--惯性权重：该参数反映了个体历史成绩对现在的影响，一般取0.5~1' \
          'cp,cg--学习因子：一般取0~0.5，此处要根据自变量的取值范围来定，并且学习因子分为个体和群体两种' \
          '位置限制：限制粒子搜索的空间，即自变量的取值范围，对于无约束问题此处可以省略。' \
          '速度限制：如果粒子飞行速度过快，很可能直接飞过最优解位置，但是如果飞行速度过慢，会使得收敛速度变慢' \
          'ub,lb--参数的上下界' \
          'pbest_x，pbest_y--每个个体在搜索过程当中的最优解和最优值' \
          'gbest_x，gbest_y--整个种群在搜索中的最优解和最优值' \
          'gbest_y_hist--记录最优解'


import numpy as np


class MyPSO:
    def __init__(self):
        self.max_iter = 500
        self.dim = 1
        self.ub = np.array([100])
        self.lb = np.array([-100])
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not False'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'
        self.pop_size = 500
        self.w = 0.6
        self.cp, self.cg = 0.5, 0.5

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop_size, self.dim))
        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop_size, self.dim))  # speed of particles
        self.Y = self.calc_obj_value()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = np.array([[np.inf]] * self.pop_size)  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration

    def calc_obj_value(self):   # 计算适应度函数
        self.Y = [10 * np.sin(5 * x) + 7 * np.cos(4 * x) for x in self.X]
        return self.Y

    def update_V(self):
        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):
        self.X += self.V
        self.X = np.clip(self.X, self.lb, self.ub)   # 将self.X中数字限制在界限之内

    def update_pbest(self):
        need_update = self.pbest_y > self.Y
        self.pbest_x = np.where(need_update, self.X, self.pbest_x)  # np.where(condition,x,y) 满足condition输出x,否则y
        self.pbest_y = np.where(need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def run(self, max_iter=None, precision=1e-3, n=20):
        """
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        """
        self.max_iter = max_iter or self.max_iter
        c = 0
        for iter_num in range(self.max_iter):
            self.update_V()
            self.update_X()
            self.calc_obj_value()
            self.update_pbest()
            self.update_gbest()
            if precision is not None:    # 整个种群的最大值-最小值 连续N次在精度之内就输出结果否则就持续到max_iter
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c += 1
                    if c > n:
                        break
                else:
                    c = 0
            print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

            self.gbest_y_hist.append(self.gbest_y)
        best_x, best_y = self.gbest_x, self.gbest_y
        return best_x, best_y


if __name__ == "__main__":
    MyPSO().run()