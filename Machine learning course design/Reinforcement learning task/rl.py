import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Step 1: Create the Ant environment
env = gym.make('Ant-v2')

# Step 2: Define the neural network architecture
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 3: Train the network parameters
def train_q_network(q_network, num_episodes=1000, gamma=0.99, epsilon=0.1):
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            state = torch.as_tensor(state, dtype=torch.float32)
            q_values = q_network(state)

            # Epsilon-greedy exploration
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            next_state = torch.as_tensor(next_state, dtype=torch.float32)

            with torch.no_grad():
                next_q_values = q_network(next_state)
                target = reward + gamma * next_q_values.max().item() * (1 - done)

            q_target = q_values.clone()
            q_target[action] = target

            loss = criterion(q_values, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

# Step 4: Compare different Q-function approximation models
def compare_models():
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]  # Use the shape of the action space for output dim

    # Model 1: QNetwork with default architecture
    model1 = QNetwork(input_dim, output_dim)

    # Model 2: QNetwork with larger hidden layers
    model2 = QNetwork(input_dim, output_dim)
    model2.fc1 = nn.Linear(input_dim, 128)
    model2.fc2 = nn.Linear(128, 128)

    # Model 3: QNetwork with additional hidden layer
    model3 = QNetwork(input_dim, output_dim)
    model3.fc1 = nn.Linear(input_dim, 64)
    model3.fc2 = nn.Linear(64, 64)
    model3.fc3 = nn.Linear(64, 64)
    model3.fc4 = nn.Linear(64, output_dim)

    models = [model1, model2, model3]

    for i, model in enumerate(models):
        print(f"Training Model {i+1}")
        train_q_network(model)

# Run the code
compare_models()
