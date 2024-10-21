import time
import numpy as np
import gymnasium as gym
from RobotInterface import RobotInterface
from gymnasium.utils.seeding import np_random
from ArucoDetectionCamera import ArucoDetectionCamera


class RealEnvironment(gym.Env):
    def __init__(self, robot_interface: RobotInterface, camera: ArucoDetectionCamera, max_actions: int):
        super(RealEnvironment, self).__init__()
        self.action_space = gym.spaces.Box(low=np.array([45, 45, 45, 45, 50, 45]),
                                           high=np.array([65, 65, 65, 65, 70, 65]),
                                           dtype=int)  # The action space
        self.observation_space = gym.spaces.Box(low=np.array([45, 45, 45, 45, 50, 45]),
                                                high=np.array([65, 65, 65, 65, 70, 65]),
                                                dtype=int)  # The observation space
        self.robot_interface = robot_interface  # The interface to the robot
        self.camera = camera  # The camera to detect the marker
        self.robot_state = None  # The state of the robot
        self.np_random = None  # Random number generator (Needed for the model to run but not used)
        self.joint_indices = [0, 1, 2, 3, 4, 5]
        self.episode_score = 0  # The score of the episode
        self.scores = []  # The scores of the episodes
        self.observation = np.zeros(6)  # The observation of the environment
        self.actions_counter = 0  # Counter for the number of actions taken
        self.max_actions = max_actions  # Max actions (algorithm steps) per each episode
        self.initial_position = None  # The initial position of the marker
        self.initial_rotation = None  # The initial rotation of the marker

    def step(self, action: list) -> tuple:
        """
        Take a step in the environment
        :param action: The action to take
        :return: The new state, the reward, whether the episode is done, and additional information
        """
        # Get the position of the marker and the time before applying the action
        previous_position, _, previous_time = self.camera.getMarkerPositionRotationAndTime()

        self.robot_interface.send_action(self.joint_indices, action)  # Send the action to the robot to be executed

        # Get the position of the marker and the time after applying the action
        current_position, current_rotation, current_time = self.camera.getMarkerPositionRotationAndTime()

        # Robot state will have all the information gathered from the robot [angles, velocities, etc.]
        robot_new_state = np.array(self.robot_interface.get_state()).squeeze()  # Get everything from the robot

        angles = robot_new_state[self.joint_indices].astype(int)  # The angles of the joints after applying the action
        print(f"Angles from arduino: {angles}")
        weight = robot_new_state[-1].astype(float)  # This will get the force measured in the arduino sketch file
        print(f"Weight from arduino: {weight} grams")
        weight = weight if weight > 0 else 0  # If the force is negative, set it to 0

        # Calculate the velocity developed while applying the action
        detected_flag = True
        if previous_position is None or current_position is None:
            dy = 0
            velocity = 0
            z_rotation = 0
            detected_flag = False
            print("Did not detect the marker")
        else:
            # Calculate the velocity of the marker
            velocity = self.camera.getMarkerVelocity(previous_position, previous_time,
                                                     current_position, current_time)

            # Flip the velocity to match the direction of movement which is towards the -x
            velocity = -velocity
            print("Velocity of the marker: {: .2f} m/s".format(velocity))

            # Calculate the distance on the y-axis from the initial position
            dy = self.camera.getMarkerDistanceY(self.initial_position, current_position)
            print("Displacement on the y-axis: {: .2f} m". format(dy))

            # Calculate the rotation on the z-axis from the initial rotation
            z_rotation = self.camera.getMarkerRotationZ(self.initial_rotation, current_rotation)
            print("Rotation on the z-axis: {: .2f} degrees". format(z_rotation))

        self.actions_counter += 1  # Increment the action counter

        dq = np.sum(np.abs(angles - self.observation))  # Calculate the difference in angles
        self.observation = angles  # Update the observation

        reward = self.calculate_reward(velocity, dy, dq, weight, z_rotation)  # Compute the reward for each step
        self.episode_score += reward  # Update the episode score

        # Check if the episode is done.
        done = self.is_done(weight, detected_flag, self.actions_counter)

        truncated = self.actions_counter == self.max_actions  # Check reached the max episode steps

        info = {}  # Additional information (Needed for the model to run but not used)

        print('---------------')
        if done:
            print(f"Total reward for current episode: {self.episode_score}\n")
            self.scores.append(self.episode_score)  # Append the score to the list of scores
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

        self.robot_state = np.array([self.robot_interface.get_state()]).squeeze()

        # Get the marker position after resetting the robot for the observation vector
        self.initial_position, self.initial_rotation, _ = self.camera.getMarkerPositionRotationAndTime()

        # Get only the moving angles
        angles = self.robot_state[self.joint_indices].astype(int)  # The angles of the joints after applying the action

        # Not required if the weight is not in the observation space
        weight = self.robot_state[-1].astype(float)  # This will get the weight measured in the arduino sketch file
        weight = weight if weight > 0 else 0  # If the weight is negative, set it to 0
        print(f"Angles from arduino: {angles}")
        print(f"Weight from arduino: {weight} grams")

        self.observation = angles

        info = {}

        self.episode_score = 0

        self.actions_counter = 0

        print('---------------')
        return self.observation, info

    def calculate_reward(self, velocity: float, y_displacement: float, dq: np.ndarray, weight: float,z_rotation: float)\
            -> float:
        # - The robot's speed: v (m/s - Reward)
        # - The weight measured by the sensor: weight (grams - Penalty)
        # - The action taken by the robot: dq (degree - Penalty)
        # - The distance from the movement axis: dy (m - Penalty)
        # - The rotation on the z-axis: z_rotation (degrees - Penalty)
        # - A small reward for not falling (r = e.g. 0.1)
        # e.g. reward = v_x - w - 0.02 * sum(a_i) + r
        old_min = 0
        old_max = 360

        new_min = 0
        new_max = -0.5
        normalized_rotation = new_min + ((z_rotation - old_min)/(old_max - old_min)) * (new_max - new_min)

        # TODO: If I decide to use this I have to set old and new min and max for the dq
        normalized_dq = new_min + ((dq - old_min)/(old_max - old_min)) * (new_max - new_min)

        return ((2 * velocity) - (0.1 * y_displacement) - (0.001 * weight) + (0.05 * self.actions_counter)
                - normalized_rotation)

    def is_done(self, weight: float, detection_flag: bool, step_counter: int) -> bool:
        # When the robot falls, the episode is done, and the environment is reset
        return weight > 200 or not detection_flag or step_counter == self.max_actions
