import gym
import numpy as np
import random
import matplotlib.pyplot as plt
# Create the Taxi environment
env = gym.make("Taxi-v3")

# Q-table (500 states x 6 actions)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 1.0    # Exploration probability
epsilon_decay = 0.999
epsilon_min = 0.1
episodes = 5000
max_steps = 100

# Training
for episode in range(episodes):
    state = env.reset()[0]   # Gymnasium API returns (state, info)
    done = False

    for step in range(max_steps):
        # Epsilon-greedy action
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-learning update rule
        best_next = np.max(q_table[next_state])
        q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * best_next)

        state = next_state
        if done:
            break

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Training finished!\n")
env = gym.make("Taxi-v3", render_mode="human")
# Play with trained agent
state = env.reset()[0]
done = False
total_reward = 0

for step in range(100):  
    env.render()
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    if done:
        break

env.close()
print("Total reward:", total_reward)
