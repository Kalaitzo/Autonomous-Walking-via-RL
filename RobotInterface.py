import time
import serial
import numpy as np


class RobotInterface:
    def __init__(self, SERIAL_PORT: str, BAUD_RATE: int = 57600):
        self.arduino = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(2)

        self.joint_angles = np.zeros(6)

    def send_action(self, servo_indices: list[int], angles: list[int]) -> None:
        if len(servo_indices) != len(angles):
            raise ValueError("The number of servo indices must match the number of angles.")

        action_parts = [f"{index}:{int(round(angle))}" for index, angle in zip(servo_indices, angles)]
        action = ",".join(action_parts) + "\n"
        self.arduino.write(action.encode('utf-8'))
        # print(f"Command sent to Arduino: {action.strip()}")

    def get_state(self) -> list:
        self.arduino.write(b"state\n")
        time.sleep(0.1)

        if self.arduino.in_waiting > 0:
            response = self.arduino.readline().decode('utf-8').rstrip()
            self.joint_angles = [int(angle) for angle in response.split(",")]
            # print(f"Received state from Arduino: {self.joint_angles}")

        return self.joint_angles

    def reset_robot(self, servo_indices: list[int], angles: list[int]) -> None:
        self.send_action(servo_indices, angles)

    def disconnect(self):
        self.arduino.close()
