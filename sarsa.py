import numpy as np
import gymnasium as gym
import time
import gridworld

class SARSAAgent:
    def __init__(self, n_states, n_act, episilon=0.1,lr=0.1,gamma=0.9):
        self.Q = np.zeros((n_states, n_act))
        self.episilon = episilon
        self.n_act = n_act
        self.lr = lr
        self.gamma = gamma

    def predict(self, state):
        Q_list = self.Q[state, :]
        # action = np.argmax(Q_list)
        action = np.random.choice(np.flatnonzero(Q_list == Q_list.max())) # 处理多个最大值
        return action
    
    def act(self, state):
        if np.random.uniform(0, 1) < self.episilon: 
            action = np.random.choice(self.n_act)
        else: 
            action = self.predict(state)       
        return action 

    def learn(self, state, action, reward, next_state, next_action, done):
        current_Q = self.Q[state, action]
       
        if done:
            target_Q = reward
        else:
            target_Q = self.gamma * self.Q[next_state, next_action] + reward
        self.Q[state, action] += self.lr * (target_Q - current_Q)

def train_episodes(env, agent, is_render):
    total_reward = 0
    state, _ = env.reset()
    action = agent.act(state)

    while True:
        next_state, reward, done, _ , info = env.step(action)
        next_action = agent.act(next_state)
        agent.learn(state, action, reward, next_state, next_action, done)


        state = next_state
        action = next_action
        total_reward += reward

        if is_render:
            env.render()
        if done:
            break
    return total_reward

def test_episodes(env, agent):
    total_reward = 0
    state, _ = env.reset()

    while True:
        action = agent.predict(state)
        next_state, reward, done, _ , info= env.step(action)
       
        state = next_state
        total_reward += reward
        env.render()
        time.sleep(0.5)
        if done:
            break
    
    return total_reward

def train(env,episodes=500,episilon=0.1,lr=0.1,gamma=0.9):
    agent = SARSAAgent(
        n_states = env.observation_space.n,
        n_act = env.action_space.n,
        lr = lr,
        gamma = gamma,
        episilon = episilon
    )

    is_render = False
    for e in range(episodes):
        ep_reward = train_episodes(env, agent, is_render)
        print('Episode %s: Reward = %.1f' % (e, ep_reward))

        if e % 50 == 0:
            is_render = True
        else:
            is_render = False
    
    test_reward = test_episodes(env, agent)
    print('test_reward = %.1f' % (test_reward))
        
if __name__ == '__main__':
    env = gym.make("CliffWalking-v1")
    # env = gridworld.CliffWalkingWapper(env)
    train(env)