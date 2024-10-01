import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

checkerboard_size = (8, 6)  # Number of inner corners per a chessboard row and column
square_size = 0.025  # Size of a square in meters (25 mm)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ... for a 3D point cloud of the checkerboard
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the images.
object_points = []  # 3d point in real world space
image_points = []  # 2d points in image plane.

images = glob.glob('img/calibration_images/*.png')

gray_shape = None

# show the images
for file_name in images:
    img = cv.imread(file_name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray_shape = gray.shape[::-1]  # Get the shape of the image (Required for the calibration)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, checkerboard_size, None)

    # If found, add object points, image points (after refining them)
    if ret:
        object_points.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        image_points.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, checkerboard_size, corners2, ret)
        cv.imshow('Image', img)
        cv.waitKey(100)

        print("Image " + file_name + " processed")

cv.destroyAllWindows()

# Calibrate the camera
ret, camera_matrix, distortion_coeffs, r_vectors, t_vectors = cv.calibrateCamera(object_points, image_points,
                                                                                 gray_shape,
                                                                                 None, None)

print("Camera matrix: ")
print(camera_matrix)

print("Distortion coefficients: ")
print(distortion_coeffs)

# Save the camera matrix and distortion coefficients
np.savez("camera_calibration_params.npz",
         camera_matrix=camera_matrix, distortion_coefficients=distortion_coeffs)

mean_error = 0
for i in range(len(object_points)):
    image_points2, _ = cv.projectPoints(object_points[i], r_vectors[i], t_vectors[i], camera_matrix, distortion_coeffs)
    error = cv.norm(image_points[i], image_points2, cv.NORM_L2) / len(image_points2)
    mean_error += error

print("total error: {}".format(mean_error / len(object_points)))
