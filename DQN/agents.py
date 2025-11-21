import numpy as np
import torch

class DQNAgent(object):
    def __init__(self, q_func, optimizer, n_act, gamma=0.9, epsilon=0.1):
        self.q_func = q_func

        self.optimizer = optimizer
        self.criterion = torch.nn.MSELoss()

        self.n_act = n_act
        self.gamma = gamma
        self.epsilon = epsilon

    def predict(self, obs):
        Q_list = self.q_func(obs)
        action = int(torch.argmax(Q_list).detach().numpy())
        return action

    def act(self, obs):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_act)
        else:
            action = self.predict(obs)
        return action

    def learn(self, obs, action, reward, next_obs, done):
        # predict_Q
        pred_Vs = self.q_func(obs)
        predict_Q = pred_Vs[action]

        # target_Q
        next_pred_Vs = self.q_func(next_obs)
        best_V = next_pred_Vs.max()
        target_Q = reward + (1 - float(done)) * self.gamma * best_V

        # 更新参数
        self.optimizer.zero_grad()
        loss = self.criterion(predict_Q, target_Q)
        loss.backward()
        self.optimizer.step()