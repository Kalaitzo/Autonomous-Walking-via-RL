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
        self.velocity = 0  # Velocity of the marker
        self.center_position = None  # Center position of the marker
        self.center_rotation = None  # Center rotation of the marker
        self.current_time = 0  # Current time

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

    def getMarkerPositionRotationAndTime(self) -> (np.ndarray, np.ndarray, float):
        """
        Get the position of the marker
        :return:
        """
        ret, frame = self.camera.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, self.aruco_dict, parameters=self.parameters)

        # cv2.imshow("Aruco Marker Detection", frame)

        if np.all(ids is not None):
            cv2.aruco.drawDetectedMarkers(gray_frame, corners, ids)

            #  Estimate the pose of the markers
            r_vecs, t_vecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.side_m,
                                                                    self.camera_matrix, self.distortion_coefficients)

            self.current_time = time.time()  # Get the current time

            # Draw the axes
            cv2.drawFrameAxes(frame,
                              self.camera_matrix, self.distortion_coefficients, r_vecs[0], t_vecs[0], self.side_m)

            self.center_position = t_vecs[0][0]

            self.center_rotation = r_vecs[0][0]

            cv2.imshow("Aruco Marker Detection", frame)

            cv2.waitKey(1)

            return self.center_position, self.center_rotation, self.current_time

        else:
            # In case the marker is not detected, return the previous position
            self.current_time = time.time()  # Get the current time
            self.center_position = None
            self.center_rotation = None
            return self.center_position, self.center_rotation, self.current_time

    def getMarkerVelocity(self,
                          previous_position: np.ndarray, previous_time: float,
                          current_position: np.ndarray, current_time: float):
        """
        Get the velocity of the marker on the x-axis
        :return: The velocity of the marker
        """
        distance_x = current_position[0] - previous_position[0]  # The distance on the x-axis (dx: meters)

        time_difference = current_time - previous_time  # The time difference (dt: seconds)

        self.velocity = distance_x / time_difference  # The velocity of the marker (v = dx/dt: m/s)

        return self.velocity

    def testCamera(self) -> None:
        ret, frame = self.camera.read()

        cv2.imshow("Camera", frame)

    @staticmethod
    def closeWindows() -> None:
        cv2.destroyAllWindows()

    @staticmethod
    def getMarkerPosition(initial_position: np.ndarray, current_position: np.ndarray):
        """
        Get the marker position with respect to the initial position
        :param initial_position:
        :param current_position:
        :return: The distance on the y-axis
        """
        return current_position - initial_position

    @staticmethod
    def getMarkerRotation(initial_rotation: np.ndarray, current_rotation: np.ndarray) -> np.ndarray:
        """
        Get the rotation with respect to the initial rotation
        :param initial_rotation:
        :param current_rotation:
        :return:
        """
        initial_rotation_matrix = cv2.Rodrigues(initial_rotation)[0]

        next_rotation_matrix = cv2.Rodrigues(current_rotation)[0]

        relative_rotation_matrix = np.dot(initial_rotation_matrix.T, next_rotation_matrix)

        relative_rotation_degrees = np.degrees(cv2.Rodrigues(relative_rotation_matrix)[0])

        return np.reshape(relative_rotation_degrees, [3,])
