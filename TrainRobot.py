import cv2
from stable_baselines3 import SAC
from RobotInterface import RobotInterface
from RealEnvironment import RealEnvironment
from ArucoDetectionCamera import ArucoDetectionCamera

# Make the connection to the robot (python - arduino)
robot = RobotInterface(SERIAL_PORT='/dev/cu.usbmodem11302')

# Create an instance of the ArucoDetectionCamera class
aruco_camera = ArucoDetectionCamera(marker_id=0, side_pixels=200, side_m=0.07,
                                    aruco_dict=cv2.aruco.DICT_6X6_250, directory="img/aruco_markers/")

# Create the necessary gym type environment that interacts with the robot
real_env = RealEnvironment(robot, aruco_camera)

# Create the learning model (SAC)
model = SAC("MlpPolicy", real_env, batch_size=2, verbose=1, learning_starts=2)

n_games = 10
time_steps = 20 * 5

key = 0

for i in range(n_games):
    # Before beginning the episode, check what the camera sees
    while True:
        aruco_camera.testCamera()

        key = cv2.waitKey(1)
        if key == 27:
            aruco_camera.closeWindows()
            break

    # This will call the step from the environment 1000 times, or until the episode is done
    # When the episode is done or before starting the steps the reset method is called
    # So the following happens:
    # - The model resets the environment
    # - The model chooses the action
    # - The action is applied to the robot by the RealEnvironment's step method
    # - The step method generates the robot_new_state, reward, done, _, and, _
    # - The model saves the new the returned values
    # - The model learns
    model.learn(total_timesteps=time_steps)

    # Save the model
    model.save("models/sac_robot")

    # Reset the key
    key = 0

    # Only require for the last n_games
    if i == n_games - 1:
        real_env.reset()  # Reset the environment so the robot is at the initial pose for the next set of games
