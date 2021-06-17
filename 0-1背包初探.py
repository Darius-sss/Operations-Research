__time__ = '2021/6/16'
__author__ = 'ZhiYong Sun'
__doc__ = '可得到一个正整数数组进行任意分隔后所有可能的和，并且可以得到具体的分隔子集' \
          '可求解 -- 判断是否可以组成某个target' \
          '可求解 -- 组成某个target的具体方式' \
          '可求解 -- 怎么分割使得相互差异最小' \
          '可求解 -- 物品价值相等的0-1背包问题'


class Partition:

    def __init__(self, nums, target):
        self.nums = sorted(nums, reverse=True)
        self.ub = sum(self.nums)
        self.target = target

    def canPartition(self):
        """
        :return: 能否分隔成target的集合，以及计算过程中的状态列表
        """

        print('start canPartition ... ')
        print('传入待分隔的数组为： ', self.nums)
        print('数组总和为：', self.ub)
        print('查询的目标值为：', self.target)

        if self.target > self.ub:
            print('目标值大于总和，请确认输入的目标值！')
            return False
        allStatus = []
        dp = [False] * (self.ub + 1)   # 定义动态数组
        dp[0] = True                  # 初始化
        for i in range(len(self.nums)):
            for j in range(self.ub, self.nums[i] - 1, -1):  # 不可以使用重复数字
                dp[j] = dp[j] or dp[j - self.nums[i]]
            allStatus.append(dp[:])
            # print(self.nums[i], dp)

        return dp[self.target], allStatus

    def getFirstTargetIndex(self, status, cur_target):
        """

        :param status: 状态列表
        :param target: 当前分隔值
        :return: 第一次出现target时的索引位置
        """

        for i in range(len(status)):
            if status[i][cur_target] is True:
                return i
        return -1

    def getPartitionIndex(self, status):
        """
        :return: 具体分隔的索引以及相应的集合
        """
        res_index, res_nums = [], []
        target = self.target
        index = self.getFirstTargetIndex(status, target)
        while target != self.nums[index]:

            num = self.nums[index]
            res_index.append(index)
            res_nums.append(num)

            target -= num
            index = self.getFirstTargetIndex(status, target)

        res_index.append(index)
        res_nums.append(self.nums[index])
        print('最终分隔后的索引为：', sorted(res_index))
        print('最终分隔后的子集合为：', sorted(res_nums))

    def main(self):
        flag, status = self.canPartition()  # 状态列表
        if not flag:
            print('该数组无法分隔成和为{0}的子集合'.format(self.target))
        else:
            self.getPartitionIndex(status)


if __name__ == '__main__':
    nums = [num for num in range(1, 101)]
    target = 1000
    Partition(nums, target).main()
