import gym 
import numpy as np
import random
#import flappy_bird_gymnasium
env=gym.make('FrozenLake-v1')
Q_tabel=np.zeros([env.observation_space.n,env.action_space.n],dtype='float')
print(Q_tabel)
alpha=0.8
gamma=0.95
num_episodes=30000
max_steps=100
epsilon=1.0
epsilon_decay=0.9999
epsilon_min=0.01
for epsiodes in range(num_episodes):
    state=env.reset()[0]
    done=False
    for steps in range(max_steps):
        if random.uniform(0,1) < epsilon:
            action=env.action_space.sample()
        else:
            action=np.argmax(Q_tabel[state])
        next_state,reward,terminate,truncation,info=env.step(action)
        done=terminate or truncation
        best_next=np.max(Q_tabel[next_state])
        Q_tabel[state,action]=(1-alpha)*Q_tabel[state,action]+alpha*(reward+gamma*best_next)
        state=next_state
        if done:
            break
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


env=gym.make('FrozenLake-v1',render_mode='human')
state = env.reset()[0]
done = False
total_reward = 0
for steps in range(max_steps):
    env.render()
    action=np.argmax(Q_tabel[state])
    state,reward,terminate,truncation,info=env.step(action)
    done=terminate or truncation
    total_reward+=reward
    if done:
        break
env.close()
print(f'total reward {total_reward}')
#print(Q_tabel)



