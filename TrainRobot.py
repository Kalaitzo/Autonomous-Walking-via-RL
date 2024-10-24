import cv2
from stable_baselines3 import SAC
from utils import plot_learning_curve
from RobotInterface import RobotInterface
from RealEnvironment import RealEnvironment
from ArucoDetectionCamera import ArucoDetectionCamera

key = 0  # The key to be pressed
attempt = 2  # The amount of learning attempts
time_steps = 500  # Number of steps to take in each training cycle
training_cycles = 3  # Number of games to play
training_cycle = 16  # The number of the training cycle that is about to start
load_time_steps = (training_cycle - 1) * time_steps  # Amount of time-steps the saved model has been trained for

models_dir = "models/"  # The models directory
plots_dir = "plots/"  # The plots directory
attempt_dir = f"attempt_{attempt}/"  # The attempt directory
model_checkpoint = f"sac_robot{load_time_steps}"  # The model checkpoint
load_path = models_dir + attempt_dir + model_checkpoint  # The complete path for the model loading

# Make the connection to the robot (python - arduino)
robot = RobotInterface(SERIAL_PORT='/dev/cu.usbmodem11301')

# Create an instance of the ArucoDetectionCamera class
aruco_camera = ArucoDetectionCamera(marker_id=0, side_pixels=200, side_m=0.07,
                                    aruco_dict=cv2.aruco.DICT_6X6_250, directory="img/aruco_markers/")

# Create the necessary gym type environment that interacts with the robot
real_env = RealEnvironment(robot, aruco_camera, max_actions=100)

load_model = True  # Whether to load a model or not
# Create the learning model (SAC)
if load_model:
    model = SAC.load(load_path, env=real_env)
else:
    model = SAC("MlpPolicy", real_env, batch_size=50, verbose=1, learning_starts=50)

for i in range(training_cycles):
    # Before beginning the episode, check what the camera sees
    while True:
        aruco_camera.testCamera()
        key = cv2.waitKey(1)
        if key == 27:  # If the key is ESC
            aruco_camera.closeWindows()  # Close the windows
            break

    # This will call the step from the environment 500 times
    # When the episode is done or before starting the steps the reset method is called
    # So the following happens:
    # - The model resets the environment
    # - The model chooses the action
    # - The action is applied to the robot by the RealEnvironment's step method
    # - The step method generates the robot_new_state, reward, done
    # - The model saves the new the returned values
    # - The model learns if the amount of time-steps is larger than the learning starts value
    model.learn(total_timesteps=time_steps, reset_num_timesteps=False)

    model_file = f"sac_robot{(i + training_cycle) * time_steps}"
    save_path = models_dir + attempt_dir + model_file
    # Save the model
    model.save(save_path)

    # Get the scores from the environment
    scores = real_env.scores

    # Plot the scores
    filename = f"Robot_Scores_{(i + training_cycle) * time_steps}.png"
    figure_file = plots_dir + attempt_dir + filename
    x = [episode + 1 for episode in range(len(scores))]
    plot_learning_curve(x, scores, figure_file)

    # Reset the key
    key = 0

# Close the windows
aruco_camera.closeWindows()
