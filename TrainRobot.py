import cv2
import csv
from utils import plot_learning_curve, plot_final_positions
from RobotInterface import RobotInterface
from RealEnvironment import RealEnvironment
from stable_baselines3 import SAC
from ArucoDetectionCamera import ArucoDetectionCamera

key = 0  # The key to be pressed
attempt = 7  # The number of learning attempts
time_steps = 500  # Number of steps to take in each training cycle
training_cycles = 3  # Number of training cycles to perform
training_cycle = 8  # The number of the training cycle that is about to start
load_time_steps = (training_cycle - 1) * time_steps  # Amount of time-steps the saved model has been trained for

models_dir = "models/"  # The models directory
scores_dir = "scores/"  # The scores directory
plots_dir = "real_robot_plots/"  # The real_robot_plots directory
attempt_dir = f"attempt_{attempt}/"  # The attempt directory
positions_dir = "positions/"  # The positions directory
model_checkpoint = f"sac_robot{load_time_steps}"  # The model checkpoint
buffer_checkpoint = f"sac_buffer{load_time_steps}"
load_path = models_dir + attempt_dir + model_checkpoint  # The complete path for the model loading
load_buffer_path = models_dir + attempt_dir + buffer_checkpoint  # The complete path for the buffer loading

# Make the connection to the robot (python - arduino)
robot = RobotInterface(SERIAL_PORT='/dev/cu.usbmodem11301')

# Create an instance of the ArucoDetectionCamera class
aruco_camera = ArucoDetectionCamera(marker_id=0, side_pixels=200, side_m=0.07,
                                    aruco_dict=cv2.aruco.DICT_6X6_250, directory="img/aruco_markers/")

# Create the necessary gym type environment that interacts with the robot
real_env = RealEnvironment(robot, aruco_camera, max_actions=100)

load_model = True  # Whether to load a model or not
if load_model:
    model = SAC.load(load_path, env=real_env)
    model.load_replay_buffer(load_buffer_path)
    model.batch_size = 256
else:
    model = SAC("MlpPolicy", real_env, batch_size=100, verbose=1, learning_starts=100)

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
    # - The model learns if the number of time-steps is larger than the learning starts value
    model.learn(total_timesteps=time_steps, reset_num_timesteps=False)

    model_file = f"sac_robot{(i + training_cycle) * time_steps}"  # The model file
    buffer_file = f"sac_buffer{(i + training_cycle) * time_steps}"  # The buffer file
    score_file = f"sac_score{(i + training_cycle) * time_steps}"  # The score file
    save_path = models_dir + attempt_dir + model_file  # The complete path for the model saving
    save_buffer_path = models_dir + attempt_dir + buffer_file  # The complete path for the buffer saving
    save_score_path = scores_dir + attempt_dir + score_file  # The complete path for the score saving

    # Get the scores from the environment
    scores = real_env.scores
    # Get the final positions from the environment
    final_positions = real_env.final_positions

    # Save the model and the replay buffer
    model.save(save_path)
    model.save_replay_buffer(save_buffer_path)
    with open(save_score_path, mode='w', newline="") as file:
        writer = csv.writer(file)
        for score in scores:
            writer.writerow([score])

    # Plot the scores
    learning_filename = f"Robot_Scores_{(i + training_cycle) * time_steps}.png"
    learning_figure_file = plots_dir + attempt_dir + learning_filename
    x = [episode + 1 for episode in range(len(scores))]
    plot_learning_curve(x, scores, learning_figure_file)

    # Plot the final positions
    if i == 2:
        positions_filename = f"Robot_Final_Positions_{(i + training_cycle) * time_steps}.png"
        positions_figure_file = plots_dir + attempt_dir + positions_dir + positions_filename
        plot_final_positions(x, final_positions, positions_figure_file)

    # Reset the key
    key = 0

# Close the windows
aruco_camera.closeWindows()
