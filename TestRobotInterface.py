import time
import numpy as np
from RobotInterface import RobotInterface

robot = RobotInterface(SERIAL_PORT='/dev/cu.usbmodem1102')

jointIndicesMapping = {"thigh_joint": 7, "leg_joint": 5, "foot_joint": 3,
                       "thigh_left_joint": 6, "leg_left_joint": 4, "foot_left_joint": 2}

try:
    robot.reset_robot()
    joint_angles = robot.get_state()
    print(f"Initial joint angles: {joint_angles}\n")

    time.sleep(1)

    for _ in range(5):
        action = np.random.randint(0, 180, 6)
        robot.send_action(list(jointIndicesMapping.values()), list(action))
        time.sleep(0.5)
        joint_angles = robot.get_state()
        print(f"Current joint angles: {joint_angles}\n")
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Program interrupted.")

finally:
    print("Disconnecting from the robot...")
    robot.disconnect()
