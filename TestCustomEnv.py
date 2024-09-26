from stable_baselines3 import SAC
from RobotInterface import RobotInterface
from RealEnvironment import RealEnvironment
from stable_baselines3.common.env_checker import check_env

# Create the robot interface that makes the communication with the robot
robot = RobotInterface(SERIAL_PORT='/dev/cu.usbmodem1102')

# Instantiate the environment
real_env = RealEnvironment(robot)

# Check the environment is suitable for the Sb3 model
# check_env(real_env)

# Create the SAC model
model = SAC("MlpPolicy", real_env, verbose=1)

# Train the model
model.learn(total_timesteps=50000)
