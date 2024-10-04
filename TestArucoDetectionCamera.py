import cv2
import time
from ArucoDetectionCamera import ArucoDetectionCamera


aruco_dict_id = cv2.aruco.DICT_6X6_250  # Get the dictionary of aruco markers id

directory = "img/aruco_markers/"  # Directory to save the image

# Create an instance of the ArucoDetectionCamera class
aruco_camera = ArucoDetectionCamera(marker_id=0, side_pixels=200, side_m=0.07,
                                    aruco_dict=aruco_dict_id, directory=directory)

while True:
    # Get the position of the marker
    previous_position, previous_time = aruco_camera.getMarkerPositionAndTime()

    time.sleep(1)  # This will be the time that it takes to apply the action on the robot (approx 1 second)

    next_position, next_time = aruco_camera.getMarkerPositionAndTime()

    # If the marker is not detected, do not calculate the velocity
    # This is only necessary for the beginning of the program.
    # Afterward, if the marker is not detected, the position stays the same as the previous one, and the velocity is 0
    if previous_position is None or next_position is None:
        velocity_x = 0
        print("Marker not detected - Velocity of the marker: {:.2f} m/s".format(velocity_x))
        continue

    # Get the velocity of the marker
    velocity_x = aruco_camera.getMarkerVelocity(previous_position, previous_time, next_position, next_time)

    # Print the velocity of the marker with 2 decimal places
    print("Velocity of the marker: {:.2f} m/s".format(velocity_x))

    key = cv2.waitKey(1)  # Wait for a key press
    if key == 27:  # If the key is the escape key
        break  # Break the loop

