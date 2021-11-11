from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from chainer import cuda
from future import standard_library

standard_library.install_aliases()

import chainer #开发深度学习算法而设计的开源框架

from chainerrl.agents import double_dqn


"""
DDQN类
act
act_and_train
"""

class DoubleDQN(double_dqn.DoubleDQN):  #继承了chainerrl.agents.double_dqn


    def act(self, state):

        #传入state状态向量，返回动作action
        """
        当train=false，也就是测试阶段，使用no_backprop_mode，那么就不会构建computational graph
        也就是不会进行反向传播、梯度下降（类似with no grad)
        chainerrl.agents.dqn里的model方法，也就是DQN网络模型
        将当前状态batch_states输入该神经网络模型，返回action_value，它是一个DiscreteActionValue对象
        """
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                action_value = self.model(self.batch_states([state], self.xp, self.phi))

                # 设置当前状态的state，保证在action_value选取动作的时候考虑一下目前已经选了的state
                # 此处不能直接写action_value.load_current_state(state)
                # 应该使用self.batch_states，保证在CPU和GPU中都能使用
                action_value.load_current_state(
                    self.batch_states([state], self.xp, self.phi)
                )  #使用DiscreteActionValue类的load_current_state方法使动作空间获取当前状态
                q = float(action_value.max.data)#最大Q值
                action = cuda.to_cpu(action_value.greedy_actions_with_state.data)[0]#采用结合state的贪婪策略选择动作 greedy_actions_with_state
        """
        结合state的贪婪策略选择动作?
        """

        # Update states
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        """
        更新Q值,乘上衰减率
        再加Qmax*(1-衰减率）
        """

        # if count == max:
        #     # print("count = {}. max = {}".format(count,max))
        #     action = len(state)

        # self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)
        return action, action_value.q_values.data.astype(np.float)

    def act_and_train(self, State, reward):


        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                action_value = self.model(
                    self.batch_states([State], self.xp, self.phi))

                # 设置当前状态的state，保证在action_value选取动作的时候考虑一下目前已经选了的state
                # 此处不能直接写action_value.load_current_state(state)
                # 应该使用self.batch_states，保证在CPU和GPU中都能使用
                action_value.load_current_state(
                    self.batch_states([State], self.xp, self.phi)
                )
                q = float(action_value.max.data)
                greedy_action = cuda.to_cpu(action_value.greedy_actions.data)[0]
                #greedy_action = cuda.to_cpu(action_value.greedy_actions_with_state.data)[ 0]
                    #greedy_actions方法 纯贪婪策略

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        # self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)

        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value)

        self.t += 1
        """
        explorer.select_action是直接继承的LinearDecayEpsilonGreedy，以lambda的概率选择greedy_action，否则随机选择
        """
        # if count == max:
        #     # print("count = {}. max = {}".format(count,max))
        #     action = len(state)

        # Update the target network
        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

        if self.last_state is not None:
            assert self.last_action is not None
            # Add a transition to the replay buffer
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=State,
                next_action=action,
                is_state_terminal=False)

        self.last_state = State
        self.last_action = action

        self.replay_updater.update_if_necessary(self.t)

        # self.logger.debug('t:%s r:%s a:%s', self.t, reward, action)

        return self.last_action, action_value.q_values.data.astype(np.float), greedy_action

