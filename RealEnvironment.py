import time
import numpy as np
import gymnasium as gym
from RobotInterface import RobotInterface
from gymnasium.utils.seeding import np_random
from ArucoDetectionCamera import ArucoDetectionCamera


class RealEnvironment(gym.Env):
    def __init__(self, robot_interface: RobotInterface, camera: ArucoDetectionCamera):
        super(RealEnvironment, self).__init__()
        self.action_space = gym.spaces.Box(low=-5, high=5, shape=(6,), dtype=float)  # The action space
        self.observation_space = gym.spaces.Box(low=0, high=180, shape=(6,), dtype=float)  # The observation space

        self.robot_interface = robot_interface  # The interface to the robot
        self.camera = camera  # The camera to detect the marker

        self.robot_state = None  # The state of the robot
        self.np_random = None  # Random number generator (Needed for the model to run but not used)
        self.joint_indices = [7, 5, 3, 6, 4, 2]

        self.target_angles = np.random.randint(0, 5, 6)  # TODO: Remove this when the reward function is implemented
        self.episode_score = 0  # The score of the episode
        self.observation = np.zeros(6)  # The observation of the environment

        self.actions_counter = 0  # Counter for the number of actions taken

    def step(self, action: list) -> tuple:
        """
        Take a step in the environment
        :param action: The action to take
        :return: The new state, the reward, whether the episode is done, and additional information
        """
        # Get the position of the marker and the time before applying the action
        previous_position, previous_time = self.camera.getMarkerPositionAndTime()

        # print("Applying action...")
        self.robot_interface.send_action(self.joint_indices, action)  # Send the action to the robot to be executed

        # Get the position of the marker and the time after applying the action
        current_position, current_time = self.camera.getMarkerPositionAndTime()

        # Robot state will have all the information gathered from the robot [angles, velocities, etc.]
        robot_new_state = np.array(self.robot_interface.get_state()).squeeze()  # Get everything from the robot

        angles = robot_new_state[self.joint_indices].astype(int)  # The angles of the joints after applying the action
        weight = robot_new_state[-1].astype(float)  # This will get the force measured in the arduino sketch file
        weight = weight if weight > 0 else 0  # If the force is negative, set it to 0

        # Calculate the velocity developed while applying the action
        detected_flag = True
        if previous_position is None or current_position is None:
            velocity = 0
            detected_flag = False
            print("Did not detect the marker")
        else:
            velocity = self.camera.getMarkerVelocity(previous_position, previous_time,
                                                     current_position, current_time)

            # Flip the velocity to match the direction of movement which is towards the -x
            velocity = -velocity

            # Calculate the distance on the y axis
            dy = self.camera.getMarkerDistanceY(previous_position, current_position)
            # print("Velocity of the marker: {:.2f} cm/s".format(velocity * 100))

        self.actions_counter += 1  # Increment the action counter

        # TODO: Create the observation from the angles, and the velocity
        # observation = np.concatenate((angles, velocity))
        self.observation = angles

        # TODO: When the reward function is established it will require:
        # - The robot's speed: v  (Reward)
        # - The force applied by the robot: f  (Penalty)
        # - The action taken by the robot: dq  (Penalty)
        # - Maybe the distance from the movement axis: dy (Penalty)
        # The reward returned is the reward for each step.
        # The episode reward is also calculated by the algorithm with sum() when the value done is True
        reward = self.calculate_reward(velocity)  # Compute the reward
        self.episode_score += reward

        # Check if the episode is done.
        # Done when: Either the weight is greater than 500 g, the marker is not detected, or 10 steps were made
        done = self.is_done(weight, detected_flag, self.actions_counter)

        truncated = False  # Check if the episode was truncated (Necessary for the model to run but not used)

        info = {}  # Additional information (Needed for the model to run but not used)

        time.sleep(0.5)  # Add a time delay so the actions are applied more smoothly

        return self.observation, reward, done, truncated, info

    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self, seed=None, options=None) -> tuple:
        """
        Reset the environment
        :return: The angles of the joints after the reset
        """
        print("Resetting the environment...")
        self.robot_interface.reset_robot()  # Reset the robot
        time.sleep(5)

        self.robot_state = np.array([self.robot_interface.get_state()]).squeeze()

        # Get only the moving angles
        angles = self.robot_state[self.joint_indices].astype(int)  # The angles of the joints after applying the action

        # Not required if the weight will not be in the observation space
        weight = self.robot_state[-1].astype(float)  # This will get the weight measured in the arduino sketch file
        weight = weight if weight > 0 else 0  # If the weight is negative, set it to 0

        self.observation = angles

        info = {}

        self.episode_score = 0

        self.actions_counter = 0

        return self.observation, info

    @staticmethod
    def calculate_reward(velocity: float) -> float:
        # TODO: Implement the actual reward function
        # The actual reward function should take in account the following:
        # - The velocity of the robot (v_x)
        # - The force applied by the robot (f)
        # - The action taken by the robot (a)
        # - A small reward for not falling (r = e.g. 0.1)
        # The force  and the action should be minimized
        # e.g. reward = v_x - f - 0.02 * sum(a_i) + r
        return velocity + 0.1

    @staticmethod
    def is_done(weight: float, detection_flag: bool, step_counter: int) -> bool:
        # TODO: Implement the done function
        # Probably the done function will check if the robot has fallen by checking force measurement
        # When the robot falls the episode is done and the robot and the episode score are reset
        return weight > 200 or not detection_flag or step_counter == 20
