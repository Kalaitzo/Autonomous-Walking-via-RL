import time
from utils import *


class ArucoDetectionCamera:
    def __init__(self, marker_id: int, side_pixels: int, side_m: float, aruco_dict: int, directory: str):
        self.camera = cv2.VideoCapture(0)  # Open the camera
        self.marker_id = marker_id  # ID of the marker
        self.side_pixels = side_pixels  # Side length of the marker in pixels (for the image)
        self.side_m = side_m  # Side length of the marker in meters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)  # Get the dictionary of aruco markers
        self.parameters = cv2.aruco.DetectorParameters()  # Parameters for the aruco marker detection
        self.directory = directory  # Directory to save the image
        self.prev_position = None  # Previous position of the marker
        self.previous_time = 0  # Previous time
        self.velocity = 0  # Velocity of the marker

        # Load the camera calibration parameters
        with np.load("camera_calibration_params.npz") as camera_calibration_params:
            self.camera_matrix = camera_calibration_params["camera_matrix"]
            self.distortion_coefficients = camera_calibration_params["distortion_coefficients"]

    def createArucoImage(self, directory: str) -> None:
        """
        Create and save the aruco marker image
        :param directory: The directory to save the image
        :return: None
        """
        createArucoMarker(self.marker_id, self.side_pixels, directory)

    def calculateCalibrationFactor(self, corners) -> float:
        """
        Calculate the calibration factor
        :param corners:
        :return:
        """

        # Calculate the side length of the marker in pixels as seen by the camera
        side_pixels = np.linalg.norm(corners[0][0][0] - corners[0][0][1])

        # Calculate the pixel-to-meter ratio
        pixel_to_meter_ratio = side_pixels / self.side_m

        return pixel_to_meter_ratio

    @staticmethod
    def getMarkerCenter(corners) -> tuple:
        marker_corners = corners[0][0]  # Get the corners of the marker

        center_x = int(np.mean(marker_corners[:, 0]))  # X coordinate of the center on the image
        center_y = int(np.mean(marker_corners[:, 1]))  # Y coordinate of the center on the image

        return center_x, center_y

    def getMarkerVelocity(self):
        """
        Get the velocity of the marker
        :return:
        """
        ret, frame = self.camera.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, self.aruco_dict, parameters=self.parameters)

        if np.all(ids is not None):
            cv2.aruco.drawDetectedMarkers(gray_frame, corners, ids)

            #  Estimate the pose of the markers
            r_vecs, t_vecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.side_m,
                                                                    self.camera_matrix, self.distortion_coefficients)

            # Draw the axes
            cv2.drawFrameAxes(frame,
                              self.camera_matrix, self.distortion_coefficients, r_vecs[0], t_vecs[0], self.side_m)

            center_position = t_vecs[0][0]  # Get the center position of the marker

            current_time = time.time()  # Get the current time

            # Calculate the velocity if the previous position is not None
            if self.prev_position is not None:
                self.velocity = calculateVelocity(self.prev_position, center_position, self.previous_time, current_time)

                displayVelocity(self.velocity, frame)

            self.prev_position = center_position
            self.previous_time = current_time

        cv2.imshow("Aruco Marker Detection", frame)

        return self.velocity

    def close(self):
        self.camera.release()
        cv2.destroyAllWindows()
        return None






