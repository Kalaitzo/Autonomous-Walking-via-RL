import cv2

camera = cv2.VideoCapture(0)

num = 23

while camera.isOpened():

    ret, frame = camera.read()

    key = cv2.waitKey(5)

    cv2.imshow("Camera", frame)

    if key == 27:  # ESC key
        break
    if key == 32:  # SPACE key
        cv2.imwrite("img/calibration_images/image" + str(num) + ".png", frame)
        print("Image " + str(num) + " saved")
        num += 1

camera.release()
cv2.destroyAllWindows()
