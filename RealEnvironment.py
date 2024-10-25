import math
import time
import numpy as np
import gymnasium as gym
from RobotInterface import RobotInterface
from gymnasium.utils.seeding import np_random
from ArucoDetectionCamera import ArucoDetectionCamera


class RealEnvironment(gym.Env):
    def __init__(self, robot_interface: RobotInterface, camera: ArucoDetectionCamera, max_actions: int):
        super(RealEnvironment, self).__init__()
        self.action_space = gym.spaces.Box(low=np.array([63, 35, 75, 30, 55, 40]),
                                           high=np.array([67, 65, 85, 40, 75, 60]),
                                           dtype=float)  # The action space
        self.observation_space = gym.spaces.Box(low=np.array([63, 35, 75, 30, 55, 40,  # Servo Angles
                                                              -1, -1, -1,  # Marker position
                                                              -180, -180, -180,  # Marker rotation
                                                              -1  # Marker velocity on the x-axis
                                                              ]),
                                                high=np.array([67, 65, 85, 40, 75, 60,  # Servo Angles
                                                               1, 1, 1,  # Marker position
                                                               180, 180, 180,  # Marker rotation
                                                               1  # Marker velocity on the x-axis
                                                               ]),
                                                dtype=float)  # The observation space
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
        self.marker_position = None  # The position of the marker with respect to the initial position
        self.marker_rotation = None  # The rotation of the marker with respect to the initial rotation

    def step(self, action: list) -> tuple:
        """
        Take a step in the environment
        :param action: The action to take
        :return: The new state, the reward, whether the episode is done, and additional information
        """
        # Get the position of the marker and the time before applying the action
        previous_position, _, previous_time = self.camera.getMarkerPositionRotationAndTime()

        self.robot_interface.send_action(self.joint_indices, action)  # Send the action to the robot to be executed
        time.sleep(0.5)  # Wait for the robot to execute the action

        # Get the position of the marker and the time after applying the action
        current_position, current_rotation, current_time = self.camera.getMarkerPositionRotationAndTime()

        # Robot state will have all the information gathered from the robot [angles, velocities, etc.]
        robot_new_state = np.array(self.robot_interface.get_state()).squeeze()  # Get everything from the robot

        angles = robot_new_state[self.joint_indices].astype(int)  # The angles of the joints after applying the action
        print(f"Angles from arduino: {angles}")
        weight = robot_new_state[-1].astype(float)  # This will get the force measured in the arduino sketch file
        print(f"Weight from arduino: {weight} grams")
        weight = weight if weight > 0 else 0  # If the force is negative, set it to 0

        rounded_action = [int(round(a_i)) for a_i in action]
        communication_error = np.any(angles != rounded_action)  # Check for arduino communication error
        if communication_error:
            print("Angles from arduino don't match the action - Reset")

        # Calculate the velocity developed while applying the action
        detected_flag = True
        if previous_position is None or current_position is None:
            self.marker_position = np.array([0, 0, 0])
            self.marker_rotation = np.array([0, 0, 0])
            x_velocity = 0
            dq = np.array([0, 0, 0, 0, 0, 0])
            detected_flag = False
            print("Did not detect the marker")
        else:
            # Calculate the velocity of the marker on the x-axis
            x_velocity = self.camera.getMarkerVelocity(previous_position, previous_time,
                                                       current_position, current_time)

            # Flip the velocity to match the direction of movement which is towards the -x
            print("Velocity of the marker: {: .2f} m/s".format(x_velocity))

            # Get the marker position with respect to the initial position
            self.marker_position = self.camera.getMarkerPosition(self.initial_position, current_position)
            y_displacement = self.marker_position[1]  # The displacement on the y-axis
            print("Displacement on the y-axis: {: .2f} cm".format(100 * y_displacement))

            # Calculate the marker rotation with respect to the initial position
            self.marker_rotation = self.camera.getMarkerRotation(self.initial_rotation, current_rotation)
            yaw = abs(self.marker_rotation[2])  # The rotation on the z-axis
            print("Rotation around the z-axis: {: .2f} degrees".format(yaw))

            dq = np.sum(np.abs(angles - self.observation[:6]))  # Calculate the difference in angles

        self.actions_counter += 1  # Increment the action counter

        self.observation = np.hstack([angles, self.marker_position, self.marker_rotation, x_velocity])

        # Compute the reward for each step
        reward = self.calculate_reward(x_velocity, self.marker_position[1], dq, weight, self.marker_position[2])
        self.episode_score += reward  # Update the episode score
        print('---------------')
        print(f"Reward for this action: {reward}")

        # Check if the episode is done.
        done = self.is_done(weight, detected_flag, self.actions_counter, communication_error)

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
        print(f"Angles from arduino: {angles}")

        # Not required if the weight is not in the observation space
        weight = self.robot_state[-1].astype(float)  # This will get the weight measured in the arduino sketch file
        print(f"Weight from arduino: {weight} grams")

        self.marker_position = np.array([0, 0, 0])  # The position of the marker with respect to the initial position

        self.marker_rotation = np.array([0, 0, 0])  # The rotation of the marker with respect to the initial rotation

        x_velocity = 0  # The velocity of the marker on the x-axis

        # Update the observation
        self.observation = np.hstack([angles, self.marker_position, self.marker_rotation, x_velocity])

        info = {}

        self.episode_score = 0

        self.actions_counter = 0

        print('---------------')
        return self.observation, info

    def calculate_reward(self, velocity: float, y_displacement: np.ndarray, dq: np.ndarray, weight: float,
                         z_rotation: np.ndarray) -> float:
        # - The robot's speed: v (m/s - Reward)
        # - The weight measured by the sensor: weight (grams - Penalty)
        # - The action taken by the robot: dq (degree - Penalty)
        # - The distance from the movement axis: dy (m - Penalty)
        # - The rotation on the z-axis: z_rotation (degrees - Penalty)
        # - A small reward for not falling (r = e.g. 0.1)
        old_min = 0
        old_max = 180

        new_min = 0
        new_max = -0.4
        normalized_rotation = new_min + ((z_rotation - old_min) / (old_max - old_min)) * (new_max - new_min)

        velocity_target = -2  # cm/s
        velocity_error = abs(100 * velocity - velocity_target)
        velocity_reward = 5 * math.exp(-0.8 * velocity_error) - 1

        # TODO: If I decide to use this I have to set old and new min and max for the dq
        normalized_dq = new_min + ((dq - old_min) / (old_max - old_min)) * (new_max - new_min)

        return (velocity_reward - y_displacement - (0.001 * weight) - normalized_rotation
                + (0.005 * self.actions_counter))

    def is_done(self, weight: float, detection_flag: bool, step_counter: int, communication_error: bool) -> bool:
        # When the robot falls, the episode is done, and the environment is reset
        return weight > 200 or not detection_flag or step_counter == self.max_actions or communication_error
