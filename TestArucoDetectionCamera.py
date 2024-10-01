import cv2
import time
from ArucoDetectionCamera import ArucoDetectionCamera


aruco_dict_id = cv2.aruco.DICT_6X6_250  # Get the dictionary of aruco markers id

directory = "img/aruco_markers/"  # Directory to save the image

# Create an instance of the ArucoDetectionCamera class
aruco_camera = ArucoDetectionCamera(marker_id=0, side_pixels=200, side_m=0.07,
                                    aruco_dict=aruco_dict_id, directory=directory)

while True:
    velocity = aruco_camera.getMarkerVelocity()  # Get the velocity of the marker

    key = cv2.waitKey(1)  # Wait for a key press
    if key == 27:  # If the key is the escape key
        break  # Break the loop

