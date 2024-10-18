import cv2
from stable_baselines3 import SAC
from utils import plot_learning_curve
from RobotInterface import RobotInterface
from RealEnvironment import RealEnvironment
from ArucoDetectionCamera import ArucoDetectionCamera

key = 0  # The key to be pressed
training_cycles = 1  # Number of games to play
time_steps = 100  # Number of steps to take in each training cycle
load_model = False  # Whether to load a model or not

# Make the connection to the robot (python - arduino)
robot = RobotInterface(SERIAL_PORT='/dev/cu.usbmodem11301')

# Create an instance of the ArucoDetectionCamera class
aruco_camera = ArucoDetectionCamera(marker_id=0, side_pixels=200, side_m=0.07,
                                    aruco_dict=cv2.aruco.DICT_6X6_250, directory="img/aruco_markers/")

# Create the necessary gym type environment that interacts with the robot
real_env = RealEnvironment(robot, aruco_camera, max_actions=100)

# Create the learning model (SAC)
if load_model:
    model = SAC.load("models/sac_robot")
else:
    model = SAC("MlpPolicy", real_env, batch_size=25, verbose=1, learning_starts=25)

for i in range(training_cycles):
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
    # model.save("models/sac_robot")

    # Get the scores from the environment
    scores = real_env.scores

    # Plot the scores
    filename = "Robot_Scores.png"
    figure_file = "plots/" + filename
    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(x, scores, figure_file)

    # Reset the key
    key = 0

# Close the windows
aruco_camera.closeWindows()
