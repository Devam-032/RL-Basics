from __future__ import annotations
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym

#Started by creating the environment and sab is for following the blackjack rules from Sutton And Barto.

env = gym.make("Blackjack-v1",sab=True)

#resetting the env to get the first Reward
done  = False
observation,info = env.reset()

#Now we set an action which needs to be taken in order to get some reward and change the state of the system.
action = env.action_space.sample()
#action = 1

#action is then executed and hence we recieve info from our env.
observation,reward,terminated,truncated,info = env.step(action)

print(f"The action taken by the agent is: {action}")
print(f"The final observation is: {observation}")
print(f"The final reward is: {reward}")
print(f"The terminate flag is: {terminated}")
print(f"The truncated flag is: {truncated}")
print(f"The env info is: {info}")

done = False
observation,info = env.reset()

class BlackJackAgent():
    def __init__(self,env,learning_rate:float,initial_epsilon:float,
                 epsilon_decay:float,final_epsilon:float,discount_factor = 0.95):
        #Q(s,a)←Q(s,a)+α*[r+γ*maxa′​Q(s′,a′)−Q(s,a)] -> temporal difference approach for q learning
        #where Q is current q value of state(s) action(a) pair.
        #α is the learning rate
        #r is the reward
        #γ is the discount factor
        #maxa′​Q(s′,a′) Estimated future reward

        #Create a dictionary where every new state automatically gets initialized with zero Q-values for all possible actions
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor

        #variables responsible for randomness are assigned
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []    

    def get_action(self,env,obs:tuple[int,int,bool])->int:
        """Return the best action with 1-epsilon probab or return a random action"""
        if np.random.rand()<self.epsilon:
            return env.action_space.sample()
        
        else:
            return int(np.argmax(self.q_values[obs]))
    
    def update(self,obs:tuple[int,int,bool],action:float,reward:float,terminated:bool,next_obs:tuple[int,int,bool]):
        """Updates q value of an action"""

        future_q = (not terminated)*(np.max(self.q_values[next_obs]))
        temporal_diff = reward + self.discount_factor*future_q - self.q_values[obs][action]

        self.q_values[obs][action] = (self.q_values[obs][action]+self.lr*temporal_diff)

        self.training_error.append(temporal_diff)
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon,self.epsilon-self.epsilon_decay)

learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon/(n_episodes/2)
final_epsilon = 0.1

agent = BlackJackAgent(env = env,learning_rate=learning_rate,initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,final_epsilon=final_epsilon,)

env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(env, obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()