import argparse
import os
import time
import copy
import pandas as pd
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import optimizers
from chainerrl import replay_buffer, explorers
import utils
from utility import env as Env, agent as DDQN, action_value as ActionValue
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

""""
    *命令行参数的定义
    argparse 模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数，然后 argparse 从 sys.argv 解析出那些参数
    ArgumentParser就是解析器，通过调用 add_argument() 方法向其中添加参数
    设备默认为CPU
"""
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--layer1-nodenum', type=int, default=64)
parser.add_argument('--lr',type=float, help='Learning rate', default=0.001)
parser.add_argument('--gamma',type=float, help='Discount factor', default=0.8)
parser.add_argument('--steps', type=int, help='steps', default=1000)
parser.add_argument('--minibatch_size', type=int, help='min batch size', default=16)
parser.add_argument('--rbuf_capacity', type=int, help='experience replay pool size', default=5 * 10 ** 3)
parser.add_argument('--replay_start_size', type=int, help='replay_start_size', default=20)
parser.add_argument('--update_interval', type=int, help='update_interval', default=10)
parser.add_argument('--MAX_EPISODE', type=int, help='MAX_EPISODE', default=10)

args = parser.parse_args()

MAX_EPISODE = args.MAX_EPISODE
gamma = args.gamma
minibatch_size = args.minibatch_size
rbuf_capacity = args.rbuf_capacity
replay_start_size = args.replay_start_size
update_interval = args.update_interval
steps = args.steps

"""
    *可变参数的定义
    数据集包含三部分数据，路径分别是:data文件夹下的disaster.csv,shelter.csv,connect.csv'
"""
# 可变参数
dataset_path = './data/ddata.xlsx'  # 原始数据集
dataset = ['disaster', 'shelter', 'connect']
MAX_EPISODE = 10
net_layers = [args.layer1_nodenum]
result_file = str('result'+'_numepisode' + str(MAX_EPISODE)+'_gamma' + str(gamma) + '_batchsize' + str(minibatch_size) + '_rbufcapacity' +str(rbuf_capacity) + '.txt')
# 每一轮逻辑如下
# 1. 初始化环境，定义S和A两个list，用来保存过程中的state和action。进入循环，直到当前这一轮完成（done == True）
# 2. 在每一步里，首先选择一个action，此处先用简单的act()代替
# 3. 接着env接收这个action，返回新的state，done和reward，当done==False时，reward=0，当done==True时，reward为避难点集合的综合评价奖励
# 4. 如果done==True，那么应该把当前的S、A和reward送到replay buffer里（replay也应该在此时进行），往replay buffer里添加时，
#    每一对state和action都有一个reward，这个reward应该和env返回的reward（也就是该模型的acc）和count有关。

episode_reward = []
loss = []

class QFunction(chainer.Chain):
    """
    obs_size：状态向量的维数、 n_actions：动作向量的维数
    Q网络输入状态，输出动作
    *根据状态数、动作数、隐藏层个数——构建线性网络模型
    输入：obs_size ——relu——隐藏层：n_hid———relu—输出层：n_actions
    """
    def __init__(self, obs_size, n_actions, n_hidden_channels=None):
        super(QFunction, self).__init__()
        if n_hidden_channels is None:
            n_hidden_channels = net_layers
        net = []
        inpdim = obs_size
        for i, n_hid in enumerate(n_hidden_channels):
            net += [('l{}'.format(i), L.Linear(inpdim, n_hid))]
            net += [('_act{}'.format(i), F.relu)]
            inpdim = n_hid
        # 构建网络模型：输入层、输出层和隐藏层
        net += [('output', L.Linear(inpdim, n_actions))]

        with self.init_scope():
            for n in net:
                if not n[0].startswith('_'):
                    setattr(self, n[0], n[1])  # 设置属性值

        self.forward = net

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        for n, f in self.forward:
            if not n.startswith('_'):
                x = getattr(self, n)(x)
            elif n.startswith('_dropout'):
                x = f(x, 0.1)
            else:
                x = f(x)

        return ActionValue.DiscreteActionValue(x)
        # 返回一个动作空间

def matplt(var,xlabel,ylabel):

    plt.figure(figsize=(20, 10))
    plt.plot(np.arange(0, MAX_EPISODE), var)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    reward_save_path = str('plot'+'_numepisode' + str(MAX_EPISODE)+'_gamma' + str(gamma) + '_batchsize' + str(minibatch_size) + '_rbufcapacity' +str(rbuf_capacity) + '.png')
    plt.savefig(reward_save_path)

def evaluate(data,eval_env, agent, current):
    """
    *单回合测试
    使用act函数选择动作

    """
    for i in range(1):
        state = eval_env.reset()
        terminal = False
        count = 0

        while not terminal:
            count += 1
            action, q = agent.act(state)
            #print('action',action)
            # 这里返回的action是一个下标值
            state, reward, terminal = eval_env.step(action,data)
            #print('terminal',terminal)
            #print(state, reward, terminal)
            if terminal:
                # 打印出每一回合的结果
                shelter_num = len(data['shelter']['id'])
                state_human = [data['shelter'].loc[i, '场所名称'] for i in range(shelter_num) if state[4][i] == 1]

                utils.log(result_file,
                          "evaluate episode:{}, reward = {}, state count = {}, state = {}"
                          .format(current, reward, len(state_human), state_human))
                agent.stop_episode()

def train_agent(data,env, agent, eval_env):

    timesum = 0
    """
       多回合训练
        terminal = False时使用act_and_train函数进行训练
        terminal = TRUE 使用stop_episode_and_train结束
    """
    for episode in range(MAX_EPISODE):
        epochstart = time.time()
        state = env.reset()
        terminal = False
        reward = 0
        while not terminal:
            action, q, ga = agent.act_and_train(
                state, reward)  # 此处action是否合法（即不能重复选取同一个指标）由agent判断。env默认得到的action合法。
            state, reward, terminal= env.step(action,data)
            if terminal:
                elapsed = time.time() - epochstart
                # 打印出每一回合的结果
                shelter_num = len(data['shelter']['id'])
                state_human = [data['shelter'].loc[i,'场所名称'] for i in range(shelter_num )if state[4][i] == 1]
                utils.log(result_file,
                           "train episode:{},loss = {}, reward = {}, state count = {},time={}, state = {}".format(episode,agent.average_loss, reward,
                                                                                                len(state_human),elapsed,state_human))
                timesum += elapsed
                agent.stop_episode_and_train(state, reward, terminal)
                episode_reward.append(reward)
                loss.append(agent.average_loss)
                if (episode + 1) % 10 == 0 and episode != 0:
                    evaluate(data,eval_env, agent, (episode + 1) / 10)
    utils.log(result_file,"The total time = {}".format(timesum))


def create_agent(env):
    """
    *创建agent
    1.初始化状态数、动作数、Qfunction
    2.初始化探索策略explorer
        epsilon:start为1，end为0.3（从非贪婪到贪婪）agent以epsilon的概率自由探索
        调用explorers.LinearDecayEpsilonGreedy 按照线性下降的贪婪策略进行动作选择
    3.初始化优化器
        使用chainer.optimizers.Adam优化器进行梯度下降，更新参数
    4.存储记忆
         使用chainer.replay_buffer构建记忆库，PrioritizedReplayBuffer优先级经验重放
    5.构建agent
        调用utility中的agent
    """
    state_size=env.state_size
    action_size = env.action_size
    q_func = QFunction(state_size, action_size)

    start_epsilon = 1.0
    end_epsilon = 0.3

    decay_steps = 100 * MAX_EPISODE/2

    explorer = explorers.LinearDecayEpsilonGreedy(start_epsilon, end_epsilon, decay_steps, env.random_action)
    opt = optimizers.Adam()
    opt.setup(q_func)
    rbuf_capacity = 5 * 10 ** 3
    minibatch_size = 16
    steps = 1000
    replay_start_size = 20
    update_interval = 10
    betasteps = (steps - replay_start_size) # update_interval
    rbuf = replay_buffer.PrioritizedReplayBuffer(rbuf_capacity, betasteps=betasteps)
    phi = lambda x: x.astype(np.float32, copy=False)  # 设置数据类型x.astype报错 dict没有astype
    # def myfunction(x):
    #     if type(x)=='dict':
    #         X=np.array(list(x.values()))
    #         return X.astype(np.float32, copy=False)
    #     return x.astype(np.float32, copy=False)
    #phi=myfunction(x)
    agent = DDQN.DoubleDQN(q_func, opt, rbuf, gamma=0.8,
                           explorer=explorer, replay_start_size=replay_start_size,
                           target_update_interval=10,  # target q网络多久和q网络同步
                           update_interval=update_interval,
                           minibatch_size=minibatch_size,
                           target_update_method='hard',
                           soft_update_tau=1e-2,
                           phi=phi,
                           gpu=args.gpu,  # 设置是否使用gpu
                           episodic_update_len=16)
    return agent
def train():
    """
    *train函数调用Env.MyEnv构造测试环境和训练环境
    调用create_agent创建agent，调用train_agent进行训练和测试
    """
    # 原始数据以Excel表格形式存储，三类数据分别存储在disaster、shelter、connect三个sheet中，使用pandas读取Excel
    # 以字典的形式存储所有数据，shelter对应的是避难所数据，disaster对应的是受灾点数据，connect对应的是路径距离数据
    Data = {}
    for file in dataset:
        Data[file] = pd.read_excel(dataset_path, sheet_name=file)

    Data['shelter']['避难人数（万人）'] = Data['shelter']['避难人数（万人）'] * 10000
    Data['disaster'] = Data['disaster'].sample(n=100, random_state=1)

    disaster_num = len(Data['disaster']['id'])
    shelter_num = len(Data['shelter']['id'])
    #Datacopy = copy.deepcopy(Data)
    # 构建训练环境，传入数据Data
    r1min = Data['shelter']['开放成本（元）'].min()
    r1max = Data['shelter']['开放成本（元）'].sum()
    distance = Data['connect']['distance']
    DISTANCE = distance.mean()
    r2min = 0
    r2max = DISTANCE * Data['disaster']['总户数'].sum()
    r3min = 0
    r3max = Data['disaster']['总户数'].sum()
    r_min_max = [r1min, r1max, r2min, r2max, r3min, r3max]
    # 构建训练环境，传入数据Data
    env = Env.MyEnv(Data,disaster_num,shelter_num,r_min_max)
    # 构建测试环境，传入数据Data,test=TRUE
    eval_env = Env.MyEnv(Data,disaster_num,shelter_num,r_min_max,test=True)

    #创建agent并进行训练与测试，传入对应的环境
    agent = create_agent(env)
    train_agent(Data,env, agent, eval_env)

    return env, agent


if __name__ == '__main__':

    train()
    matplt(episode_reward,'episode','reward')
    matplt(loss,'episode','loss')







