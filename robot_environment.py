from gym import Env
from gym.spaces import Box
import numpy as np
import random


class RobotEnv(Env):
    def __init__(self, standingRewardWeight=1, walkingRewardWeight=1):
        # Observation space will be:
        # Force calculate by the force sensor that the robot is holding on (1)
        # Speed of the robot (1)
        # Position of the robot (1)
        # Each servo joint angle (10)
        self.observation_space = Box(low=-1000000, high=1000000, shape=(12,))

        self.action_space = Box(low=-30, high=30, shape=(10,))  # The servo angles
        # We set the action space to be between 0 and 30 degrees for each joint because there is no need for bigger
        # angles. The robot will not be able to move if the angles are bigger than 30 degrees.

        force = self.force  # The force calculated by the force sensor (should be around 0 at the beginning)
        velocity = self.velocity  # The speed of the robot (should be around 0 at the beginning)

        self.observation = np.array([90, 90, 90, 0, 0, 90, 90, 90, 0, 0, force, velocity])  # Initial state of the robot
        self.steps = 1000  # Number of steps the robot can take before the episode ends

        # Reward weights
        self.standingRewardWeight = standingRewardWeight  # Weight of the standing reward
        self.walkingRewardWeight = walkingRewardWeight  # Weight of the walking reward

    @property
    def force(self):
        # TODO: Write the function that gets the force from the force sensor
        return None

    @property
    def velocity(self):
        # TODO: Write the function that calculates the speed of the robot
        return None

    def _get_observation(self):
        # TODO: I think here I need to get the previous observation from the buffer in order to use it for the action I
        #  just wrote this code to run the program without errors. IT IS WRONG AND NEEDS TO BE CHANGED
        servo_angles = self.observation[:10]  # The servo angles
        force = np.array(self.observation[10])  # The force calculated by the force sensor
        velocity = np.array(self.observation[11])  # The speed of the robot

        # Maybe something like
        # servo_angles = self.observation[:10]  # The servo angles
        # force = np.array(self.force)  # The force calculated by the force sensor
        # velocity = np.array(self.velocity) # The speed of the robot

        # The observation will be the servo angles, the force, and the velocity in a single 1x12 array
        observation = np.concatenate((servo_angles, force, velocity), axis=None)
        return observation

    def step(self, action):
        # Update the joint part of the observation of the robot
        self.observation[:10] = action  # Update the servo angles of the robot
        # TODO: Find out how is the action going to be applied to the robot servos

        self.steps -= 1  # Reduce the number of steps the robot can take

        observation = self._get_observation()  # Get the new observation of the robot

        force = observation[-2]  # TODO: Get the force from the force sensor (depends on the indexing I will use)
        standing_reward = self.standingRewardWeight * force  # It will depend on the force sensor

        velocity = observation[-1]  # TODO: Calculate the speed of the robot (depends on the indexing I will use)
        walking_reward = self.walkingRewardWeight * velocity  # It will depend on the speed of the robot

        reward = standing_reward + walking_reward  # The reward will be the sum of the standing and walking rewards

        # Check if the episode is done
        done = self.steps <= 0 or force > 1  # The episode is done if the robot has taken all the steps or has "fallen."
        # TODO: Tune the values of the force and the steps (steps=1000, force>1 ?)

        info = {
            "force": force,  # The force calculated by the force sensor
            "velocity": velocity  # The speed of the robot
        }

        return self.observation, reward, done, info

    def reset(self):
        self.observation = np.array([90, 90, 90, 0, 0, 90, 90, 90, 0, 0, 0, 0])  # Reset the state of the robot
        self.steps = 1000  # Reset the number of steps the robot can take
        return self.observation

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
