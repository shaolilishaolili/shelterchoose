import argparse
import os
import time

import pandas as pd
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import optimizers
from chainerrl import replay_buffer, explorers

import utils
from utility import env as Env, agent as DDQN, action_value as ActionValue
# linux命令行使用，复制以下命令即可执行 nohup python start.py --result-file result.txt  --gpu 1 --layer1-nodenum 64 --layer2-nodenum
# 32>training_log.txt 2>&1 &

""""
    *命令行参数的定义
    argparse 模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数，然后 argparse 从 sys.argv 解析出那些参数
    ArgumentParser就是解析器，通过调用 add_argument() 方法向其中添加参数
    '--result-file'默认将结果存储到'result.txt'
    设备默认为CPU
    隐藏层单元个数：--layer1-nodenum 64 ，layer2-nodenum 32
"""
parser = argparse.ArgumentParser()
parser.add_argument('--result-file', type=str, default='result.txt')
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--layer1-nodenum', type=int, default=64)
parser.add_argument('--layer2-nodenum', type=int, default=32)
args = parser.parse_args()

"""
    *可变参数的定义
    数据集包含三部分数据，路径分别是:data文件夹下的disaster.csv,shelter.csv,connect.csv'
    候选避难场所的总数目为 3
    训练的轮数为1000
    神经网络共两层，每层的节点数由命令行参数确定
"""
# 可变参数
dataset_path = './data/data.xlsx'  # 原始数据集
dataset = ['disaster', 'shelter', 'connect']
MAX_EPISODE = 1000
net_layers = [args.layer1_nodenum, args.layer2_nodenum]

# 每一轮逻辑如下
# 1. 初始化环境，定义S和A两个list，用来保存过程中的state和action。进入循环，直到当前这一轮完成（done == True）
# 2. 在每一步里，首先选择一个action，此处先用简单的act()代替
# 3. 接着env接收这个action，返回新的state，done和reward，当done==False时，reward=0，当done==True时，reward为避难点集合的综合评价奖励
# 4. 如果done==True，那么应该把当前的S、A和reward送到replay buffer里（replay也应该在此时进行），往replay buffer里添加时，
#    每一对state和action都有一个reward，这个reward应该和env返回的reward（也就是该模型的acc）和count有关。


episode_reward = []


# evaluate_reward = []


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
            # net += [('norm{}'.format(i), L.BatchNormalization(n_hid))]
            net += [('_act{}'.format(i), F.relu)]
            # net += [('_dropout{}'.format(i), F.dropout)]
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


def evaluate(eval_env, agent, current):
    """
    *单回合测试
    使用act函数选择动作

    """
    for i in range(1):
        state = eval_env.reset()
        terminal = False
        count = 0
        while not terminal:
            action, q = agent.act(state)

            # 这里返回的action是一个下标值
            count += 1
            state, reward, terminal = eval_env.step(action)

            if terminal:
                state_human = [i + 1 for i in range(len(state)) if state[i] == 1]
                #evaluate_reward.append(reward)
                utils.log(args.result_file,
                          "evaluate episode:{}, reward = {}, state count = {}, state = {}"
                          .format(current, reward, len(state_human), state_human))

                agent.stop_episode()


def train_agent(env, agent):
    """
    *多回合训练
    terminal = False时使用act_and_train函数进行训练
    terminal = TRUE 使用stop_episode_and_train结束
    每10个回合训练一次：evaluate（）
    """
    for episode in range(MAX_EPISODE):

        state = env.reset()
        terminal = False
        reward = 0
        count = 0

        while not terminal:
            action, q, ga = agent.act_and_train(
                state, reward)  # 此处action是否合法（即不能重复选取同一个指标）由agent判断。env默认得到的action合法。
            count += 1
            state, reward, terminal = env.step(action)

            if terminal:
                # 打印出每一回合的结果
                state_human = [i + 1 for i in range(len(state)) if state[i] == 1]
                utils.log(args.result_file,
                          "train episode:{}, reward = {}, state count = {}, state = {}".format(episode, reward,
                                                                                               len(state_human),
                                                                                               state_human))

                agent.stop_episode_and_train(state, reward, terminal)
                episode_reward.append(reward)
                # if (episode + 1) % 10 == 0 and episode != 0:
                #      evaluate(eval_env, agent, (episode + 1) / 10)


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
    state_size = env.state_size
    action_size = env.action_size
    q_func = QFunction(state_size, action_size)

    start_epsilon = 1.
    end_epsilon = 0.5
    decay_steps = state_size * MAX_EPISODE / 2  # epsilon线性下降，从1到0.3，这个下降的过程共经过decay_steps步
    #设置decay_steps为state_size * MAX_EPISODE/2是保持agent能在前500回合（MAX_EPISODE / 2）进行探索，
    # 前500回合的步数最多为 state_size * MAX_EPISODE/2，因为每一回合至多state_size步（选择所有的避难所，每选一个是一步）
    explorer = explorers.LinearDecayEpsilonGreedy(start_epsilon, end_epsilon, decay_steps, env.random_action)

    opt = optimizers.Adam()
    opt.setup(q_func)

    rbuf_capacity = 5 * 10 ** 3
    minibatch_size = 16

    steps = 1000
    replay_start_size = 20
    update_interval = 10
    betasteps = (steps - replay_start_size) // update_interval
    rbuf = replay_buffer.PrioritizedReplayBuffer(rbuf_capacity, betasteps=betasteps)

    phi = lambda x: x.astype(np.float32, copy=False)  # 设置数据类型

    agent = DDQN.DoubleDQN(q_func, opt, rbuf, gamma=0.99,
                           explorer=explorer, replay_start_size=replay_start_size,
                           target_update_interval=10,  # target q网络多久和q网络同步
                           update_interval=update_interval,
                           phi=phi, minibatch_size=minibatch_size,
                           target_update_method='hard',
                           soft_update_tau=1e-2,
                           gpu=args.gpu,  # 设置是否使用gpu
                           episodic_update_len=16)  # episodic_update=False报错，没有这个参数
    return agent


def train():
    """
    *train函数调用Env.MyEnv构造测试环境和训练环境
    调用create_agent创建agent，调用train_agent进行训练和测试

    """
    """
    数据读取
    """
    # 原始数据以Excel表格形式存储，三类数据分别存储在disaster、shelter、connect三个sheet中，使用pandas读取Excel
    # 以字典的形式存储所有数据，shelter对应的是避难所数据，disaster对应的是受灾点数据，connect对应的是路径距离数据
    Data = {}
    for file in dataset:
        Data[file] = pd.read_excel(dataset_path, sheet_name=file)
    # print('打印数据集：')
    # print('**************shelter**************\n', Data['shelter'])
    # print('**************disaster**************\n', Data['disaster'])
    # print('****************connect*****************\n', Data['connect'])


    # 构建训练环境，传入数据Data
    env = Env.MyEnv(Data)
    # 构建测试环境，传入数据Data,test=TRUE
    eval_env = Env.MyEnv(Data, test=True)

    #创建agent并进行训练与测试，传入对应的环境
    agent = create_agent(env)
    train_agent(env, agent)

    return env, agent


if __name__ == '__main__':

    train()




"""
    # 用于计算本次训练中最大的准确率以及平均准确率
    max_reward = max(episode_reward)
    average_reward = 0
    for i in range(len(episode_reward)):
        average_reward += episode_reward[i]
    average_reward = average_reward / len(episode_reward)
    # 评估结果
    max_evaluate_reward = max(evaluate_reward)
    average_evaluate_reward = 0
    for i in range(len(evaluate_reward)):
        average_evaluate_reward += evaluate_reward[i]
    average_evaluate_reward = average_evaluate_reward / len(evaluate_reward)
"""
