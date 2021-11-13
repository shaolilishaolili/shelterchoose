import copy
import random
import numpy as np
from utility import newreward

# 自定义环境，agent探索的环境为候选避难场所的集合，使用状态向量state表征
# 环境模块要实现的功能有：更新状态（step）、获取奖励（get_reward)、重置环境（rest）
class MyEnv:
    def __init__(self, data,disaster_num,shelter_num,r_min_max,test=False):
        self.shelter_num=shelter_num
        self.disaster_num=disaster_num
        self.data = data
        self.dict = {}
        #self.test = test  # 是否为测试
        self.state_size=6*shelter_num  #状态的尺寸是3个向量、3个数值"拉平"
        self.action_size = shelter_num + 1  # 动作数也就是避难所个数加1（因为可以选择不避难）
        self.step_size = disaster_num # 步数等于受灾点的个数
        self.r_min_max = r_min_max
        self.state = {}  # 初始状态,采用字典的形式，存放各避难所的剩余容量和分配情况
        self.state['id'] = np.zeros(self.shelter_num, np.int) #当前受灾点的id
        self.state['need'] = np.zeros(self.shelter_num, np.int) #当前受灾点是否需要避难
        self.state['population'] = np.zeros(self.shelter_num, np.int)#当前受灾点的人数
        self.state['location'] = np.zeros(shelter_num, np.float)  # 当前受灾点到每一个避难所的距离
        self.state['is_open'] = np.zeros(shelter_num, np.int) #避难所是否开放
        self.state['r_capacity'] = copy.deepcopy(self.data['shelter']['避难人数（万人）'].to_numpy())
        self.step_count = 0  # 当前步数
        self.done = False  # 是否终止选择
        self.alloc=np.zeros((disaster_num,shelter_num),np.float)
        self.DISTANCE=data['connect']['distance'].mean()

    def step(self,action_index,data):  # 测试时传入的action是一个下标值，它表示被选中的避难所的索引，当等于shelter_num时表示不避难
        self.step_count += 1
        if self.step_count-1 == self.step_size: #如果步数达到受灾点个数，终止该回合，计算奖励
            reward = self.get_reward()  # 计算奖励
            self.done = True
            return np.array(list(self.state.values())), reward, self.done
        else:  # 否则，进行状态更新，
            self.state['id'][0]=data['disaster']['id'].to_numpy()[self.step_count-1]#更新id
            self.state['need'][0] = data['disaster']['isDisaster'].to_numpy()[self.step_count-1]#更新是否需要避难
            self.state['location'][0]=data['disaster']['lng'].to_numpy()[self.step_count-1]
            self.state['location'][1]=data['disaster']['lat'].to_numpy()[self.step_count-1]
            if (self.state['need'][0] == 0): #不需要避难
                if action_index == self.shelter_num: #选的就是不避难
                    reward = 0
                else:   #选了避难所
                    reward = -1
                    return np.array(list(self.state.values())), reward, self.done
            else:#如果需要避难就按照约束修改相应的isopen和alloc
                d=data['disaster']['总户数'].to_numpy()
                self.state['population'][0] = d[self.step_count-1]
                if action_index != self.shelter_num:
                    #但在进行分配前需要先考虑目前的剩余容量是否满足要求
                    distance = data['connect'][(data['connect']['disasterid'] == self.state['id'][0])]
                    distance = distance[(distance['shelterid'] == action_index + 1)]['distance'].item()
                    # 容量约束与距离约束同时满足时，进行分配and self.state['distance'][action_index]<self.DISTANCE
                    if self.state['r_capacity'][action_index] >= d[self.step_count-1] and distance < self.DISTANCE:
                        self.state['is_open'][action_index] = 1  #被选中的避难所开放
                        self.state['r_capacity'][action_index] -= d[self.step_count-1] #更新容量
                        self.alloc[self.step_count-1][action_index] = d[self.step_count-1] #记录分配信息

                    else:    #容量不满足要求，返回一个负的奖励
                        reward = -1
                        return np.array(list(self.state.values())), reward, self.done
                reward = 0
        return np.array(list(self.state.values())), reward, self.done

    def get_reward(self):
        temp = [str(x) for x in self.state]  # 状态向量里的每个值转为字符串
        temp = '.'.join(temp)

        if temp not in self.dict:  # 如果状态不在字典里，就调用newreward里的get_reward（）
            reward = newreward.get_reward(self.state,self.data,self.r_min_max,self.alloc)
            self.add_dict(reward)  # 将state-reward放入字典
        else:
            reward = self.dict.get(temp)
        return reward

    def add_dict(self, reward):
        temp = [str(x) for x in self.state]
        temp = '.'.join(temp)
        self.dict[temp] = reward

    # 使用字典的好处：不用重复计算reward

    def reset(self):
        self.dict = {}
        self.state_size = 6 * self.shelter_num # 状态的尺寸是3个向量、3个数值"拉平"
        self.action_size = self.shelter_num + 1  # 动作数也就是避难所个数加1（因为可以选择不避难）
        self.step_size = self.disaster_num  # 步数等于受灾点的个数
        self.state = {}  # 初始状态,采用字典的形式，存放各避难所的剩余容量和分配情况
        self.state['id'] = np.zeros(self.shelter_num , np.int)  # 当前受灾点的id
        self.state['need'] = np.zeros(self.shelter_num , np.int)  # 当前受灾点是否需要避难
        self.state['population'] = np.zeros(self.shelter_num , np.int)  # 当前受灾点的人数
        self.state['location'] = np.zeros(self.shelter_num, np.float)  # 当前受灾点到每一个避难所的距离
        self.state['is_open'] = np.zeros(self.shelter_num , np.int)  # 避难所是否开放

        # self.state['r_capacity']=data['shelter']['避难人数（万人）'].to_numpy()#避难所的剩余容量
        self.state['r_capacity'] = copy.deepcopy(self.data['shelter']['避难人数（万人）'].to_numpy())
        self.count = 0  # 当前已经选取的避难场所数
        self.step_count = 0  # 当前步数
        self.done = False  # 是否终止选择
        self.alloc = np.zeros((self.disaster_num, self.shelter_num), np.float)
        return np.array(list(self.state.values()))

    def random_action(self):  # 在环境中还定义了如何随机地进行动作选择，也就是产生随机数，范围从0到action_size
        action = random.randint(0, self.action_size - 1)
        return action
