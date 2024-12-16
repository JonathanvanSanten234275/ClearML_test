from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import OT2Env
import gymnasium as gym
import numpy as np

# instantiate your custom environment
env = OT2Env(render=False) # modify this to match your wrapper class

# Assuming 'env' is your wrapped environment instance
check_env(env)

num_episodes = 5

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    step = 0

    while not done:
        # Take a random action from the environment's action space
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if truncated:
            done = True
            print(f"truncated")
            break
        elif terminated:
            done = True
            print(f"terminated")
            break

        print(f"Episode: {episode + 1}, Step: {step + 1}, Action: {action}, Reward: {reward}")

        step += 1
        if done:
            print(f"Episode finished after {step} steps. Info: {info}")
            break