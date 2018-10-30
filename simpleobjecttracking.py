import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while 1:

    _, frame = cap.read()

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    lower_blue = np.array([150, 150, 150])
    upper_blue = np.array([255, 255, 255])

    mask = cv2.inRange(RGB, lower_blue, upper_blue)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:



        break



cv2.destroyAllWindows()
