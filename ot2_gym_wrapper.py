import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import random
import time

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32)

        # keep track of the number of steps
        self.steps = 0
        self.robotId = 0

    def reset(self, seed=None):
        # being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # set a random goal position for the agent, consisting of x, y, and z coordinates within the working area (you determined these values in the previous datalab task)
        self.goal_position = [random.uniform(-0.1870, 0.2531), random.uniform(-0.1705, 0.2209), random.uniform(0.1197, 0.2209)]

        # Call the environment reset function
        observation = self.sim.reset(num_agents=1)
        self.robotId = int(list(observation.keys())[-1][-1])

        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        v_abs = 0
        
        pipette_coords = np.array(self.sim.get_pipette_position(self.robotId))
        goal_coords = np.array(self.goal_position)
        observation = np.concatenate([pipette_coords, goal_coords, [v_abs]]).astype(np.float32)

        info = {}

        # Reset the number of steps
        self.steps = 0
        self.set_step_2_stop = False
        self.close2dish = False
        self.close2goal = False
        self.steps_taken_2_stop = None
        self.stopped_at_goal = False
        self.reward_at_dish = 0
        self.reward_at_target = 0
        self.reward_at_stop = 0
        self.init_d_goal = 0

        return (observation, info)

    def step(self, action):
        terminated = False
        truncated = False
        d_goal_max = 0.5975980337986396
        # Execute one time step within the environment
        # since we are only controlling the pipette position, we accept 3 values for the action and need to append 0 for the drop action
        action = np.append(action, 0)

        # Call the environment step function
        observation = self.sim.run([action]) # Why do we need to pass the action as a list? Think about the simulation class.
        self.robotId = int(list(observation.keys())[-1][-1])

        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        v_joint_x = observation[f'robotId_{self.robotId}']['joint_states']['joint_0']['velocity']
        v_joint_y = observation[f'robotId_{self.robotId}']['joint_states']['joint_1']['velocity']
        v_joint_z = observation[f'robotId_{self.robotId}']['joint_states']['joint_2']['velocity']
        v_abs = np.abs(v_joint_x) + np.abs(v_joint_y) + np.abs(v_joint_z)
        
        pipette_coords = np.array(self.sim.get_pipette_position(self.robotId))
        goal_coords = np.array(self.goal_position)
        observation = np.concatenate([pipette_coords, goal_coords, [v_abs]]).astype(np.float32)

        d_goal = np.linalg.norm(observation[:3] - observation[3:6])

        # Calculate the reward, this is something that you will need to experiment with to get the best results
        if self.steps == 0:
            self.init_d_goal = d_goal

        reward = ((self.init_d_goal / d_goal_max) - (d_goal / d_goal_max))*100
        reward -= self.steps / 10
        base_reward = reward
        
        # next we need to check if the if the task has been completed and if the episode should be terminated
        # To do this we need to calculate the distance between the pipette position and the goal position and if it is below a certain threshold, we will consider the task complete. 
        # What is a reasonable threshold? Think about the size of the pipette tip and the size of the plants.
        if d_goal < 0.01:
            if self.set_step_2_stop == False:
                self.step_2_stop = self.steps
                self.set_step_2_stop = True

            if d_goal < 0.005 and observation[6] < 0.007:
                terminated = True
                self.steps_taken_2_stop = self.steps - self.step_2_stop
                self.reward_at_stop =  max(0, 60 - 0.3 * self.steps_taken_2_stop)
                if self.reward_at_stop == 0:
                    reward += 1
                    self.stopped_at_goal = True
                else:
                    reward += self.reward_at_stop
                    self.stopped_at_goal = True
            else:
                terminated = False

        
        # next we need to check if the episode should be truncated, we can check if the current number of steps is greater than the maximum number of steps
        if self.steps == self.max_steps:
            truncated = True
        else:
            truncated = False

        info = {'d-goal': d_goal,
                'speed':observation[6],
                 'checks':{f"base reward:": base_reward,
                           f"stopped at: {self.steps_taken_2_stop}": self.reward_at_stop}}

        # increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()
