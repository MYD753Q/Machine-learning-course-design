import gym

# 创建Walker2d环境
env = gym.make('Walker2d-v3')

# 执行初始化动作
observation = env.reset()

# 渲染环境并执行动作
for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # 这里使用随机动作作为示例
    observation, reward, done, info = env.step(action)
