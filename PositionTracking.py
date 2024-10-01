import time
from utils import *

marker_id = 0  # ID of the marker
side_m = 0.07  # Side length of the marker in cm (7 cm) - Measured with a ruler

# Load the camera calibration parameters
with np.load("camera_calibration_params.npz") as camera_calibration_params:
    cam_matrix = camera_calibration_params["camera_matrix"]
    distortion_coefficients = camera_calibration_params["distortion_coefficients"]

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # Get the dictionary of aruco markers
parameters = cv2.aruco.DetectorParameters()  # Parameters for the aruco marker detectionD

camera = cv2.VideoCapture(0)  # Open the camera

prev_position = None  # Previous position of the marker
prev_time = 0  # Previous time

while True:
    ret, frame = camera.read()  # Read the frame from the camera

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

    corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)  # Detect the aruco markers

    # If the markers are detected, find their center coordinates
    if np.all(ids is not None):
        cv2.aruco.drawDetectedMarkers(gray_frame, corners, ids)  # Draw the outlines of the detected markers

        # Estimate the pose of the markers
        r_vecs, t_vecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, side_m, cam_matrix, distortion_coefficients)

        center_position = t_vecs[0][0]  # Get the center position of the marker

        current_time = time.time()  # Get the current time

        # Calculate the velocity if the previous position is not None
        if prev_position is not None:
            velocity = calculateVelocity(prev_position, center_position, prev_time, current_time)  # Calculate the v_x

            displayVelocity(velocity, frame)  # Display the velocity information on the frame

        cv2.drawFrameAxes(frame, cam_matrix, distortion_coefficients, r_vecs[0], t_vecs[0], side_m)  # Draw the axes

        # Update the previous position and time
        prev_position = center_position
        prev_time = current_time

    cv2.imshow("Aruco Marker Detection", frame)  # Display the frame

    key = cv2.waitKey(1)  # Wait for a key press
    if key == 27:  # If the key is the escape key
        break  # Break the loop

camera.release()  # Release the camera
cv2.destroyAllWindows()  # Close the windows
