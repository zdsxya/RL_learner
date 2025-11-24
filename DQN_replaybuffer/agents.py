import numpy as np
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import torchUtils

class DQNAgent(object):
    def __init__(self, q_func, optimizer, replay_buffer, replay_start_size, n_act, 
                 gamma=0.9, epsilon=0.1, batch_size=32):
        self.global_step = 0
        self.q_func = q_func
        
        self.rb = replay_buffer
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size

        self.optimizer = optimizer
        self.criterion = torch.nn.MSELoss()

        self.n_act = n_act
        self.gamma = gamma
        self.epsilon = epsilon

    def predict(self, obs):
        obs = torch.FloatTensor(obs)

        Q_list = self.q_func(obs)
        action = int(torch.argmax(Q_list).detach().numpy())
        return action

    def act(self, obs):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_act)
        else:
            action = self.predict(obs)
        return action
    
    def learn_batch(self, batch_obs, batch_action, batch_reward, batch_next_obs, batch_done):
        # predict_Q
        pred_Vs = self.q_func(batch_obs)
        action_onehot = torchUtils.one_hot(batch_action, self.n_act)
        predict_Q = (pred_Vs * action_onehot).sum(dim = 1)

        # target_Q
        # next_pred_Vs = self.q_func(batch_next_obs)
        # best_V = next_pred_Vs.max()
        # target_Q = batch_reward + (1 - batch_done) * self.gamma * best_V
        target_Q = batch_reward + (1 - batch_done) * self.gamma * self.q_func(batch_next_obs).max(1)[0]


        # 更新参数
        self.optimizer.zero_grad()
        loss = self.criterion(predict_Q, target_Q)
        loss.backward()
        self.optimizer.step()

    def learn(self, obs, action, reward, next_obs, done):
        self.global_step += 1
        self.rb.append((obs, action, reward, next_obs, done))
        if len(self.rb) > self.replay_start_size and self.global_step % self.rb.num_steps == 0:
            self.learn_batch(*self.rb.sample(self.batch_size))