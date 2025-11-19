import gymnasium as gym
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

class QLearningAgent:
    def __init__(self, n_states, n_act, epsilon=0.1, lr=0.1, gamma=0.9):
        self.Q = np.zeros((n_states, n_act))
        self.n_act = n_act
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma

    def predict(self, state):
        Q_list = self.Q[state, :]
        action = np.random.choice(np.flatnonzero(Q_list == Q_list.max()))
        return action

    def act(self, state):
        if np.random.uniform(0,1) < self.epsilon:
            action = np.random.choice(self.n_act)
        else:
            action = self.predict(state)
        return action
    
    def learn(self, state, action, reward, next_state, done):
        current_Q = self.Q[state, action]
       
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_state, :])

        self.Q[state, action] += self.lr * (target_Q - current_Q)

def train_episodes(env, agent):
    total_reward = 0
    state, _ = env.reset()
    action = agent.act(state)

    while True:
        next_state, reward, done, _ , info = env.step(action)
        next_action = agent.act(next_state)
        agent.learn(state, action, reward, next_state, done)

        state = next_state
        action = next_action
        total_reward += reward
        # time.sleep(0.5)

        if done:
            break
    return total_reward


def test_episodes(env, agent):
    total_reward = 0
    state, _ = env.reset()

    while True:
        action = agent.predict(state)
        next_state, reward, done, _ , info = env.step(action)
        state = next_state
        total_reward += reward
        if done:
            break

    return total_reward

def train(env, episodes=500, epsilon=0.1, lr=0.1, gamma=0.9):
    agent = QLearningAgent(
        n_states = env.observation_space.n,
        n_act = env.action_space.n,
        lr = lr,
        gamma = gamma,
        epsilon = epsilon
    )

    for e in range(episodes):
        ep_reward = train_episodes(env, agent)
        print('Episode %s: Reward = %.1f' % (e, ep_reward))
    
    test_reward = test_episodes(env, agent)
    print('test_reward = %.1f' % test_reward)

if __name__ == '__main__':
    env = gym.make("CliffWalking-v1")
    train(env)