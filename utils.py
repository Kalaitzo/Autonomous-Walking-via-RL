import numpy as np
import matplotlib.pyplot as plt
import pybullet as p


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
