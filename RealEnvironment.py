import gymnasium as gym
import numpy as np
from RobotInterface import RobotInterface
from gymnasium.utils.seeding import np_random


class RealEnvironment(gym.Env):
    def __init__(self, robot_interface: RobotInterface):
        super(RealEnvironment, self).__init__()
        self.action_space = gym.spaces.Box(low=0, high=5, shape=(6,), dtype=float)
        self.observation_space = gym.spaces.Box(low=-0, high=5, shape=(6,), dtype=float)
        self.robot_interface = robot_interface
        self.robot_state = None
        self.np_random = None
        self.joint_indices = [7, 5, 3, 6, 4, 2]
        self.target_angles = np.random.randint(0, 5, 6)
        print(f"Target angles: {self.target_angles}")

    def step(self, action: list) -> tuple:
        """
        Take a step in the environment
        :param action: The action to take
        :return: The new state, the reward, whether the episode is done, and additional information
        """
        # print("Applying action...")
        self.robot_interface.send_action(self.joint_indices, action)  # Send the action to the robot to be executed

        # Uncomment this if necessary
        # time.sleep(0.5)

        # Robot state will have all the information gathered from the robot [angles, velocities, etc.]
        robot_new_state = np.array(self.robot_interface.get_state()).squeeze()  # Get the new state after the action

        # TODO: When the reward function is established it will require only the new robot state
        reward = self.calculate_reward(robot_new_state, self.target_angles)  # Compute the reward
        # print(f"Reward: {reward}")

        done = self.is_done(robot_new_state)  # Check if the episode is done

        truncated = False  # Check if the episode was truncated

        info = {}

        return robot_new_state, reward, done, truncated, info

    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self, seed=None, options=None) -> tuple:
        """
        Reset the environment
        :return: The angles of the joints after the reset
        """
        # print("Resetting the environment...")
        self.robot_interface.reset_robot(self.joint_indices, [90] * 6)

        self.robot_state = np.array([self.robot_interface.get_state()]).squeeze()

        info = {}

        return self.robot_state, info

    @staticmethod
    def calculate_reward(robot_state: np.ndarray, target_angles: np.ndarray) -> float:
        # TODO: Implement the actual reward function
        # Check if the robot state is close to the target angles
        # Reward should be higher if the robot is closer to the target angles
        # Reward should be lower otherwise
        squared_error = np.sum((robot_state - target_angles) ** 2)

        max_error = 5 * 5 * 6

        reward = -squared_error / max_error

        return reward

    @staticmethod
    def is_done(robot_state: np.ndarray) -> bool:
        # TODO: Implement the done function
        done = False if robot_state is None else True
        return done
