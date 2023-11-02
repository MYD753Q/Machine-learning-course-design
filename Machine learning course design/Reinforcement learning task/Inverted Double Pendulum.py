import gym
import torch
import torch.nn as nn
import numpy as np


# 创建环境
env = gym.make('InvertedDoublePendulum-v2')

# 定义 Q 函数逼近模型
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 初始化 Q 网络和目标网络
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
q_network = QNetwork(state_size, action_size)
target_network = QNetwork(state_size, action_size).eval()


# 定义优化器和损失函数
optimizer = torch.optim.Adam(q_network.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 定义超参数
num_episodes = 1000
epsilon = 0.1
discount_factor = 0.99

# 辅助函数：根据 epsilon-greedy 策略选择动作
def select_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        q_values = q_network(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return torch.argmax(q_values).item()

# 经验回放缓冲区
replay_buffer = []

# 训练智能体
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        replay_buffer.append((state, action, reward, next_state, done))

        state = next_state

    # 更新目标网络
    target_network.load_state_dict(q_network.state_dict())

    # 打印奖励
    print(f"Episode {episode}: Total Reward = {total_reward}")

    # 在每个 episode 结束后清空 replay_buffer
    replay_buffer = []

# 关闭环境
env.close()
