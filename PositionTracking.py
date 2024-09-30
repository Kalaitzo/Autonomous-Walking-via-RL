import time
from utils import *

marker_id = 0  # ID of the marker
side_pixels = 200  # Side length of the marker in pixels
side_cm = 0.071  # Side length of the marker in cm (7.1 cm) - Measured with a ruler

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # Get the dictionary of aruco markers

parameters = cv2.aruco.DetectorParameters()  # Parameters for the aruco marker detectionD

directory = "img/aruco_markers/"  # Directory to save the image

marker_image = createArucoMarker(marker_id, side_pixels, directory)  # Create and save the aruco marker image

camera = cv2.VideoCapture(0)  # Open the camera

pixel_to_meter_ratio = side_pixels / side_cm  # Pixel-to-meter ratio

prev_position = None  # Previous position of the marker
previous_time = 0  # Previous time

# Parameters for filtering
positions = []  # Store recent position measurements
filter_size = 5  # Number of samples to average

while True:
    ret, frame = camera.read()  # Read the frame from the camera

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

    corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)  # Detect the aruco markers

    # If the markers are detected, find their center coordinates
    if np.all(ids is not None):
        cv2.aruco.drawDetectedMarkers(gray_frame, corners, ids)  # Draw the outlines of the detected markers

        pixel_to_meter_ratio = calculateCalibrationFactor(corners, side_cm)  # Calculate the pixel-to-meter ratio

        center_position = getMarkerCenter(corners)  # The position of the marker on the image not in the real world

        # Filter the position
        # positions.append(center_position)
        # if len(positions) > filter_size:
        #     positions.pop(0)
        # center_position = np.mean(positions, axis=0).astype(int)

        current_time = time.time()  # Get the current time as(t_k: s)

        # Calculate the velocity if the previous position is not None
        if prev_position is not None:
            velocity = calculateVelocity(prev_position, center_position, previous_time, current_time,
                                         pixel_to_meter_ratio)  # Calculate the velocity

            cv2.circle(frame, center_position, 5, (0, 0, 255), -1)  # Draw the center of the marker

            drawFrameAxis(center_position[0], center_position[1], frame)  # Draw the x-y axis

            displayVelocity(velocity, frame)  # Display the velocity information

        # Update the previous position and time
        prev_position = center_position
        previous_time = current_time

    cv2.imshow("Frame", frame)  # Display the frame

    key = cv2.waitKey(1)  # Wait for a key press
    if key == 27:  # If the key is the escape key
        break  # Break the loop

camera.release()  # Release the camera
cv2.destroyAllWindows()  # Close the windows
