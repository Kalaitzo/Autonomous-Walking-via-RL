import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import os
import cv2
import math


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
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


def calculateVelocity(prev_position: tuple, position: tuple, previous_time: float, current_time: float,
                      pixel_to_meter_ratio: float) -> float:
    distance_x = np.linalg.norm(np.array(position[0]) - np.array(prev_position[0]))  # The pixel distance

    real_distance = distance_x / pixel_to_meter_ratio  # Convert the distance to meters (dx: m)

    time_difference = current_time - previous_time  # Calculate the time difference (dt: s)

    velocity = real_distance / time_difference  # Calculate the velocity (v = dx/dt: m/s)

    return velocity


def getMarkerCenter(corners) -> tuple:
    marker_corners = corners[0][0]  # Get the corners of the marker

    center_x = int(np.mean(marker_corners[:, 0]))  # X coordinate of the center on the image
    center_y = int(np.mean(marker_corners[:, 1]))  # Y coordinate of the center on the image

    return center_x, center_y


def drawFrameAxis(cX: int, cY: int, frame: np.ndarray) -> None:
    # Draw the x and y axes lines at the marker's center
    line_length = 200  # Length of the axis lines in pixels
    x_axis_start = (cX - line_length // 2, cY)  # Start point for x-axis
    x_axis_end = (cX + line_length // 2, cY)  # End point for x-axis
    y_axis_start = (cX, cY - line_length // 2)  # Start point for y-axis
    y_axis_end = (cX, cY + line_length // 2)  # End point for y-axis

    # Draw x-axis (horizontal) in red
    cv2.line(frame, x_axis_start, x_axis_end, (255, 0, 0), 2)
    # Draw y-axis (vertical) in blue
    cv2.line(frame, y_axis_start, y_axis_end, (0, 255, 0), 2)

    return None


def displayVelocity(velocity: float, frame: np.ndarray) -> None:
    # Show the horizontal linear velocity value in the top left corner
    cv2.putText(frame,
                f"Velocity: {velocity:.2f} m/s or {velocity * 100:.2f} cm/s",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    return None


def calculateCalibrationFactor(corners, real_marker_length: float) -> float:
    # Calculate the pixel distance between the two corners of the marker
    marker_side = np.linalg.norm(corners[0][0][0] - corners[0][0][1])

    # Calculate the pixel-to-meter ratio
    pixel_to_meter_ratio = marker_side / real_marker_length

    return pixel_to_meter_ratio











