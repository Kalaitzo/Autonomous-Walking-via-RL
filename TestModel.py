import numpy as np
import cv2
from RobotInterface import RobotInterface
from RealEnvironment import RealEnvironment
from stable_baselines3 import SAC
from ArucoDetectionCamera import ArucoDetectionCamera

load_path = "models/attempt_5/sac_robot10000"
# load_buffer_path = "models/attempt_5/sac_buffer10000"

# Make the connection to the robot (python - arduino)
robot = RobotInterface(SERIAL_PORT='/dev/cu.usbmodem11301')

# Create an instance of the ArucoDetectionCamera class
aruco_camera = ArucoDetectionCamera(marker_id=0, side_pixels=200, side_m=0.07,
                                    aruco_dict=cv2.aruco.DICT_6X6_250, directory="img/aruco_markers/")

# Create the necessary gym type environment that interacts with the robot
real_env = RealEnvironment(robot, aruco_camera, max_actions=100)

model = SAC.load(load_path, env=real_env)
# model.load_replay_buffer(load_buffer_path)
# model.batch_size = 256

n_episodes = 3
episode_rewards = []

for episode in range(n_episodes):

    while True:
        aruco_camera.testCamera()
        key = cv2.waitKey(1)
        if key == 27:  # If the key is ESC
            aruco_camera.closeWindows()  # Close the windows
            break

    observation = real_env.reset()
    observation = np.array(observation[0])
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, _, _ = real_env.step(action)
        total_reward += reward

    episode_rewards.append(total_reward)

    key = 0

# Calculate average reward
avg_reward = np.mean(episode_rewards)
print(f"Average Reward over {n_episodes} episodes: {avg_reward}")
