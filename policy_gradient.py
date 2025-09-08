import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

learning_rate = 0.01
gamma = 0.99   
policy = PolicyNetwork(state_size, action_size)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
def run_episode(env, policy):
    state = env.reset()[0] 
    rewards = []
    log_probs = []
    done = False
    
    while not done:
        state_tensor = torch.FloatTensor(state)
        probs = policy(state_tensor)
        m = Categorical(probs)
        action = m.sample()
        
        log_prob = m.log_prob(action)
        state, reward, done, _, _ = env.step(action.item())
        
        log_probs.append(log_prob)
        rewards.append(reward)
        
    return log_probs, rewards
def compute_returns(rewards, gamma):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    return returns
def update_policy(log_probs, returns):
    loss = []
    for log_prob, R in zip(log_probs, returns):
        loss.append(-log_prob * R)
    loss = torch.stack(loss).sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
num_episodes = 1000

for episode in range(num_episodes):
    log_probs, rewards = run_episode(env, policy)
    returns = compute_returns(rewards, gamma)
    update_policy(log_probs, returns)
    
    if (episode+1) % 50 == 0:
        print(f'Episode {episode+1}, total reward: {sum(rewards)}')

        

env = gym.make('CartPole-v1',render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy = PolicyNetwork(state_size, action_size)
test_episodes = 10
total_rewards = []

with torch.no_grad():  # disable gradient calculation
    for episode in range(test_episodes):
        state = env.reset()[0]  # Gymnasium returns (obs, info)
        done = False
        episode_reward = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state)
            probs = policy(state_tensor)
            action = torch.argmax(probs).item()  # greedy action
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
        total_rewards.append(episode_reward)
        print(f"Test Episode {episode+1}: Total Reward = {episode_reward}")

avg_reward = sum(total_rewards) / len(total_rewards)
print(f"Average Test Reward over {test_episodes} episodes: {avg_reward}")

env.close()
