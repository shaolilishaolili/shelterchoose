import random
import numpy as np
from utility import newreward


# from utility import reward as cls


# action space 中的最后一个动作为终止

# 自定义环境，agent探索的环境为候选避难场所的集合，使用状态向量state表征
# 环境模块要实现的功能有：更新状态（step）、获取奖励（get_reward)、重置环境（rest）
class MyEnv:
    def __init__(self, data, test=False):
        self.data = data
        self.dict = {}
        self.test = test  # 是否为测试
        self.state_size = len(self.data['shelter']['id'])  # 状态数也就是候选避难场所的总个数
        self.action_size = self.state_size + 1  # 动作数也就是状态数加1（因为有一个终止动作）
        self.state = [0 for _ in range(self.state_size)]  # 初始状态向量的每个元素都为0，说明没有选中任何避难场所
        self.count = 0  # 当前已经选取的避难场所数
        self.done = False  # 是否终止选择
        # if self.test:
        #     #self.action_size = state_size  # 测试时选够max个指标，不要包含终止动作
        # else:
        #     self.action_size = state_size + 1  # 包含一个终止动作

    """
    step为状态更新函数，根据St,At，得到St+1
    更新的规则是：当action_index在(0,state_size-1)之间，执行选择动作，把State向量中对应下标的元素置为1
    当action_index=state_size，执行终止动作，将done置为Ture
    每次更新都要根据目前选择的避难所集合，获取reward,
        reward的获取规则是：先将当前的状态向量传入self.get_reward,如果能在字典中查询到对应的奖励，就直接返回该奖励
        如果字典中没有相关记录，就调用newreward模块中的get_reward函数，并把state-reward存入字典，以备下次查询
    """

    def step(self, action_index):  # 测试时传入的action是一个下标值，所以就是index
        if action_index == self.state_size:  # 如果动作超出最大值,终止
            self.done = True
        else:
            self.state[action_index] = 1  # 否则，设置状态向量为1
            self.count += 1

        reward = self.get_reward()  # 从字典里查询状态对应的奖励
        return np.array(self.state), reward, self.done  #

    def get_reward(self):
        temp = [str(x) for x in self.state]  # 状态向量里的每个值转为字符串
        temp = '.'.join(temp)
        reward = self.dict.get(temp, -0.00000001)  # 如果当前State在字典里，就返回对应的reward数值，否则返回-0.00000001
        if reward == -0.00000001:  # 如果状态不在字典里，就调用newreward里的get_reward（）
            reward = newreward.get_reward(self.state, self.data)
            self.add_dict(reward)  # 将state-reward放入字典
        return reward

    def add_dict(self, reward):
        temp = [str(x) for x in self.state]
        temp = '.'.join(temp)
        self.dict[temp] = reward

    # 使用字典的好处：不用重复计算reward

    def reset(self):
        self.state = [0 for _ in range(self.state_size)]  # 初始状态向量的每个元素都为0
        self.count = 0  # 当前已经选取的避难所个数
        self.done = False  # 是否终止
        return np.array(self.state)

    def random_action(self): #在环境中还定义了如何随机地进行动作选择，也就是产生随机数，范围从0到state_size,但action=state_size表示final
        while True:
            action = random.randint(0, self.action_size - 1)# 如果选择的action对应的避难所已被选中，继续产生随机数

            if action == self.action_size - 1 or self.state[action] == 0:  #直到选到一个未被选择的避难所或选到final
                break
        return action
