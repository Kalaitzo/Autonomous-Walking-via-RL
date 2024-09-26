import numpy as np
from gym import spaces
from pybullet_envs.gym_locomotion_envs import Walker2DBulletEnv
from utils import get_angles


class CustomWalker2DPyBulletEnv(Walker2DBulletEnv):
    def __init__(self):
        super(CustomWalker2DPyBulletEnv, self).__init__()

        # Modify the observation space
        self.episode_reward = 0
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 6),  # Change lower bounds as needed
            high=np.array([np.inf] * 6),  # Change upper bounds as needed
            dtype=np.float32
        )

    def step(self, action):
        # Call the parent class step method to get the original observations
        obs, reward, done, info = super().step(action)

        robot_id = self.robot.objects[0]
        joint_ids = [joint.jointIndex for joint in self.robot.jdict.values()]
        angles = get_angles(robot_id, joint_ids)

        # Transform the observation to fit the new observation space
        transformed_obs = angles

        self.episode_reward += reward
        if done:
            print("Episode reward: ", self.episode_reward)
            self.episode_reward = 0

        return transformed_obs, reward, done, info

    def reset(self):
        # Reset the environment and get the initial observation
        super().reset()
        robot_id = self.robot.objects[0]
        joint_ids = [joint.jointIndex for joint in self.robot.jdict.values()]
        angles = get_angles(robot_id, joint_ids)
        transformed_obs = angles
        return transformed_obs

