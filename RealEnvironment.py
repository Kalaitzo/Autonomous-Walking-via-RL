import numpy as np
import gymnasium as gym
from RobotInterface import RobotInterface
from gymnasium.utils.seeding import np_random
from ArucoDetectionCamera import ArucoDetectionCamera


class RealEnvironment(gym.Env):
    def __init__(self, robot_interface: RobotInterface, camera: ArucoDetectionCamera):
        super(RealEnvironment, self).__init__()
        self.action_space = gym.spaces.Box(low=0, high=5, shape=(6,), dtype=float)
        self.observation_space = gym.spaces.Box(low=0, high=180, shape=(6,), dtype=float)

        self.robot_interface = robot_interface
        self.camera = camera

        self.robot_state = None
        self.np_random = None
        self.joint_indices = [7, 5, 3, 6, 4, 2]  # Change this to the actual channel indices on the PCA9685 board

        self.target_angles = np.random.randint(0, 5, 6)

        self.episode_score = 0

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
        robot_new_state = np.array(self.robot_interface.get_state()).squeeze()  # This will get the angles and the force

        # TODO: Calculate the robot's speed
        # Will need to add the velocity to the robot_new_state (observation)
        # velocity = self.camera.getMarkerVelocity()

        # TODO: When the reward function is established it will require:
        # - The robot's speed
        # - The force applied by the robot
        # - The action taken by the robot
        reward = self.calculate_reward(robot_new_state, self.target_angles)  # Compute the reward
        self.episode_score += reward

        # TODO: When the done function is established it will require:
        # - The force applied by the robot (to check if the robot is considered fallen)
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

        self.episode_score = 0

        return self.robot_state, info

    @staticmethod
    def calculate_reward(robot_state: np.ndarray, target_angles: np.ndarray) -> float:
        # TODO: Implement the actual reward function
        # The actual reward function should take in account the following:
        # - The velocity of the robot (v_x)
        # - The force applied by the robot (f)
        # - The action taken by the robot (a)
        # - A small reward for not falling (r = e.g. 0.1)
        # The force  and the action should be minimized
        # e.g. reward = v_x - f - 0.02 * sum(a_i) + r

        # This was for testing purposes
        squared_error = np.sum((robot_state - target_angles) ** 2)

        max_error = 5 * 5 * 6

        reward = -squared_error / max_error

        return reward

    @staticmethod
    def is_done(robot_state: np.ndarray) -> bool:
        # TODO: Implement the done function
        # Probably the done function will check if the robot has fallen by checking force measurement
        # When the robot falls the episode is done and the robot and the episode score are reset
        done = False if robot_state is None else True
        return done
