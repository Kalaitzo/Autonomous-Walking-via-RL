import time
import serial
import numpy as np


class RobotInterface:
    def __init__(self, SERIAL_PORT: str, BAUD_RATE: int = 4800):
        self.arduino = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(2)

        self.robot_state = np.zeros(11)

    def send_action(self, servo_indices: list[int], angles: list[int]) -> None:
        if len(servo_indices) != len(angles):
            raise ValueError("The number of servo indices must match the number of angles.")

        action_parts = [f"{index}:{int(round(angle))}" for index, angle in zip(servo_indices, angles)]
        action = ",".join(action_parts) + "\n"
        self.arduino.write(action.encode('utf-8'))
        print(f"Command sent to Arduino: {action.strip()}")

    def get_state(self) -> list:
        self.arduino.write(b"state\n")
        time.sleep(0.8)

        if self.arduino.in_waiting > 0:
            response = self.arduino.readline().decode('utf-8').rstrip()
            self.robot_state = response.split(",")

        return self.robot_state

    def reset_robot(self) -> None:
        self.arduino.write(b"reset\n")
        time.sleep(3)

        if self.arduino.in_waiting > 0:
            response = self.arduino.readline().decode('utf-8').rstrip()
            print(f"Reset confirmation from Arduino: {response}")

    def disconnect(self):
        self.arduino.close()
