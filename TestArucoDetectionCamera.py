import cv2
import time

import numpy as np

from ArucoDetectionCamera import ArucoDetectionCamera


aruco_dict_id = cv2.aruco.DICT_6X6_250  # Get the dictionary of aruco markers id

directory = "img/aruco_markers/"  # Directory to save the image

# Create an instance of the ArucoDetectionCamera class
aruco_camera = ArucoDetectionCamera(marker_id=0, side_pixels=200, side_m=0.07,
                                    aruco_dict=aruco_dict_id, directory=directory)

# if ket is the space key, then the initial position is set
while True:
    initial_position, initial_rotation, _ = aruco_camera.getMarkerPositionRotationAndTime()
    key = cv2.waitKey(1)  # Wait for a key press
    if key == 32:  # If the key is the space key
        break
print("Initial position: ", initial_position)
print("Initial rotation: ", initial_rotation)

while True:
    next_position, next_rotation, _ = aruco_camera.getMarkerPositionRotationAndTime()
    # If the marker is not detected, do not calculate the velocity
    # This is only necessary for the beginning of the program.
    # Afterward, if the marker is not detected, the position stays the same as the previous one, and the velocity is 0
    if next_position is None:
        print("Marker not detected - Displacement is 0 - Rotation is 0")
        continue

    relative_rotation_degrees = aruco_camera.getMarkerRotation(initial_rotation, next_rotation)

    # Print the velocity of the marker with 2 decimal places
    print("=====================================")
    print("X-axis displacement: {:.2f} cm".format(100 * (next_position[0] - initial_position[0])))
    print("Y-axis displacement: {:.2f} cm".format(100 * (next_position[1] - initial_position[1])))
    print("Z-axis displacement: {:.2f} cm".format(100 * (next_position[2] - initial_position[2])))
    print("=====================================\n")
    print("X-axis rotation: {:.2f} degrees".format(relative_rotation_degrees[0]))
    print("Y-axis rotation: {:.2f} degrees".format(relative_rotation_degrees[1]))
    print("Z-axis rotation: {:.2f} degrees".format(relative_rotation_degrees[2]))
    print("=====================================\n")

    time.sleep(1)  # Sleep for 1 second

    key = cv2.waitKey(1)  # Wait for a key press
    if key == 27:  # If the key is the escape key
        break  # Break the loop

