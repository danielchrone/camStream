import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while 1:

    _, frame = cap.read()

    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_green = np.array([38, 82, 92])
    upper_green = np.array([75, 255, 255])



    lower_red = np.array([169, 100, 100])
    upper_red = np.array([190, 255, 255])

    maskBlue = cv2.inRange(HSV, lower_blue, upper_blue)
    maskGreen = cv2.inRange(HSV, lower_green, upper_green)
    maskRed = cv2.inRange(HSV, lower_red, upper_red)
    mask = maskBlue + maskGreen + maskRed
    res = cv2.bitwise_and(frame, frame, mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    #cv2.imshow('maskREd', maskRed)
    #cv2.imshow('maskGreen', maskGreen)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:



        break



cv2.destroyAllWindows()
