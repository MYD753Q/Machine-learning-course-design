import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from ddpg import Actor, Critic, DDPG, ReplayBuffer  # 导入你的自定义类

# 构建环境
env = gym.make('Walker2d-v3')  # 使用Walker2d环境

# 参数设置
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
actor_lr = 1e-4
critic_lr = 1e-3
gamma = 0.99
tau = 0.001
buffer_size = 1000000
batch_size = 64
exploration_noise = 0.1

# 初始化神经网络和优化器
actor = Actor(state_dim, action_dim)
actor_target = Actor(state_dim, action_dim)
critic = Critic(state_dim, action_dim)
critic_target = Critic(state_dim, action_dim)
actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

# 初始化DDPG智能体和重播缓冲区
ddpg = DDPG(actor, actor_target, critic, critic_target, actor_optimizer, critic_optimizer, gamma, tau,
            exploration_noise)
replay_buffer = ReplayBuffer(buffer_size)

# 训练DDPG智能体
num_episodes = 500
reward_history = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = ddpg.select_action(state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        ddpg.train()

    reward_history.append(total_reward)
    print(f"Episode: {episode + 1}, Reward: {total_reward}")

# 可视化奖励曲线
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DDPG Training Progress")
plt.show()

# 测试DDPG智能体
test_episodes = 10
test_rewards = []

for _ in range(test_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = ddpg.select_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

    test_rewards.append(total_reward)

avg_test_reward = np.mean(test_rewards)
print(f"Average Test Reward: {avg_test_reward}")

# 关闭环境
env.close()
