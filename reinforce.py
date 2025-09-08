import gym 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
class policyNetwork(nn.Module):
	def __init__(self,action_space,observation_space):
		super(policyNetwork,self).__init__()
		self.f1=nn.Linear(observation_space,128)
		self.f2=nn.Linear(128,action_space)
	def forward(self,x):
		f1=self.f1(x)
		return F.softmax(self.f2(f1),dim=-1)

env=gym.make('CartPole-v1')
observation_space=env.observation_space.shape[0]
action_space=env.action_space.n
network=policyNetwork(action_space,observation_space)
optimizer=optim.Adam(network.parameters(),lr=0.001)
for episodes in range(1):
	state=env.reset()[0]
	log_prob=[]
	rewards=[]
	done=False
	while not done:
		state_tensor=torch.tensor(state)
		print(f'this is the state:{state} ')
		probs=network(state_tensor)
		print(f'this is the probs:{probs}')
		m=Categorical(probs)
		
		print(f'this is m:{m}')
		action=m.sample()
		#print(m.sample())
		print(f'this is action {action}')
		log_prob.append(m.log_prob(action))
		state,reward,done,_,_=env.step(action.item())
		rewards.append(reward)
	returns=[]
	gamma=0.9
	g=0
	for r in reversed(rewards):
		g=r+gamma*g
		returns.insert(0,g)
	returns=torch.tensor(returns)
	returns=(returns-returns.mean())/(returns.std()+1e-9)
	loss=[]
	for log_prob,g in zip(log_prob,returns):
		loss.append(-log_prob*g)
	loss=torch.stack(loss).sum()
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print(f"Episode {episodes}, Total Reward: {sum(rewards)}")

'''total_reward=[]
env=gym.make('CartPole-v1',render_mode='human')
for i in range(10):
	state=env.reset()[0]
	done=False
	ep_reward=0
	while not done:
		state_tensor=torch.FloatTensor(state)
		probs=network(state_tensor)
		m=Categorical(probs)
		action=m.sample().item()
		state,reward,done,_,_=env.step(action)
		ep_reward+=reward
	total_reward.append(ep_reward)
	print(f"Episode {i+1}: Reward = {ep_reward}")'''



