import agents, modules, replay_buffer
import gymnasium as gym
import torch

class TrainManager():
    def __init__(self, env, episodes=500, lr=0.001, gamma=0.9, epsilon=0.1, memory_size = 2000,
                 replay_start_size = 200, batch_size = 32, num_steps = 4):
        self.env = env
        n_act = env.action_space.n
        n_obs = env.observation_space.shape[0]
        q_func = modules.MLP(n_obs, n_act)
        optimizer = torch.optim.AdamW(q_func.parameters(), lr=lr)
        rb = replay_buffer.ReplayBuffer(memory_size, num_steps)

        self.agent = agents.DQNAgent(
            q_func = q_func,
            optimizer = optimizer,
            n_act = n_act,
            replay_buffer = rb,
            replay_start_size = replay_start_size,
            gamma = gamma,
            epsilon = epsilon
        )
        self.episodes = episodes

    def train_episode(self):
        total_reward = 0
        obs, info = self.env.reset()

        while True:
            action = self.agent.act(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done  = terminated or truncated

            self.agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            if done:
                break
        return total_reward
    
    def test_episode(self):
        total_reward = 0
        obs, info = self.env.reset()

        while True:
            action = self.agent.predict(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done  = terminated or truncated

            obs = next_obs
            total_reward += reward
            self.env.render()
            if done:
                break
        return total_reward
    
    def train(self):
        for e in range(self.episodes):
            ep_reward = self.train_episode()
            print('Episode %s: reward = %.1f' % (e, ep_reward))

            if e % 100 == 0:
                test_reward = self.test_episode()
                print('Test reward = %.1f' % (test_reward))

if __name__ == '__main__':
    env1 = gym.make("CartPole-v1")
    tm = TrainManager(env1)
    tm.train()