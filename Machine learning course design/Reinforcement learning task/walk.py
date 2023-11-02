import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 构建强化学习环境
env = gym.make('Walker2d-v3')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


# 定义多层感知机（MLP）网络
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义深度强化学习代理
class Agent:
    def __init__(self, Q_network, learning_rate=0.001, gamma=0.99):
        self.Q_network = Q_network
        self.target_network = QNetwork()  # 用于双重Q学习
        self.target_network.load_state_dict(self.Q_network.state_dict())
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=learning_rate)
        self.gamma = gamma

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return env.action_space.sample()  # 随机选择动作
        state = torch.tensor(state, dtype=torch.float32)
        q_values = self.Q_network(state)
        action = torch.argmax(q_values).item()
        return action

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)  # 强制将动作变为整数
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        q_value = self.Q_network(state)[action]
        next_q_value = reward + self.gamma * torch.max(self.target_network(next_state))
        loss = torch.nn.functional.mse_loss(q_value, next_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.Q_network.state_dict())


# 训练参数
num_episodes = 1000
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
update_target_freq = 10
reward_history = []

# 初始化代理
q_network = QNetwork()
agent = Agent(q_network)

# 训练主循环
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.train(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            break

    reward_history.append(total_reward)

    # 更新epsilon以实现探索与利用的平衡
    epsilon = max(epsilon * epsilon_decay, min_epsilon)

    # 更新目标网络
    if episode % update_target_freq == 0:
        agent.update_target_network()

    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

# 绘制奖励曲线
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()
