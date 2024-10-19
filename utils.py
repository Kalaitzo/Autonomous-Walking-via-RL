import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import os
import cv2


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title(f'Running average of previous {len(scores)} scores')
    plt.savefig(figure_file)


def keep_same_angles(action, indexes):
    for index in indexes:
        action[index] = 0
    return action


def get_angles(robot_id, joint_ids):
    angles = []  # Array for the new angles
    for joint_id in joint_ids:
        angle = p.getJointState(robot_id, joint_id)[0]  # Get a joint angle (This is in radians probably)
        angles.append(angle)

    angles = np.array(angles)  # Convert to a numpy array
    angles_deg = np.rad2deg(angles)  # Convert to degrees
    return angles_deg


def reformatObservation(observation: list, angles: list) -> list:
    # Reformat the observation
    observation = [angles]
    return observation


def createArucoMarker(marker_id: int, side_pixels: int, directory: str) -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # Get the dictionary of aruco markers

    # Create an image from the dictionary
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, side_pixels)  # Image of specific marker

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_name = os.path.join(directory, f"marker_{marker_id}.jpg")  # File to save the image

    # Save the image
    cv2.imwrite(file_name, marker_image)

    return marker_image


def calculateVelocity(prev_position: tuple, position: tuple, previous_time: float, current_time: float):
    distance_x = np.array(position[0]) - np.array(prev_position[0])  # The distance in the x direction

    time_elapsed = current_time - previous_time  # Time elapsed between the two positions

    velocity = distance_x / time_elapsed  # Calculate the velocity (v = d/t: m/s)

    return velocity


def displayVelocity(velocity: float, frame: np.ndarray) -> None:
    # Show the horizontal linear velocity value in the top left corner
    cv2.putText(frame,
                f"Velocity: {velocity:.2f} m/s or {velocity * 100:.2f} cm/s",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    return None











