import gymnasium as gym

env = gym.make("CartPole-v1")
obs,info = env.reset()
obs,reward,terminated,truncated,info = env.step(action=0)
print(f"Obs: {obs},Reward: {reward},Terminated: {terminated},Truncated: {truncated},Info: {info}")