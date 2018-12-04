import cv2
import numpy as np

kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))
cap = cv2.VideoCapture("Test3.mp4")
font = cv2.FONT_HERSHEY_SIMPLEX

textBlue = "Shoulder"
textRed = "Elbow"
textGreen = "Hand"

while 1:
    _, frame = cap.read()
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 80])
    upper_blue = np.array([110, 255, 255])

    lower_green = np.array([38, 82, 92])
    upper_green = np.array([75, 255, 255])

    lower_red = np.array([169, 100, 100])
    upper_red = np.array([190, 255, 255])

    maskBlue = cv2.inRange(HSV, lower_blue, upper_blue)
    maskGreen = cv2.inRange(HSV, lower_green, upper_green)
    maskRed = cv2.inRange(HSV, lower_red, upper_red)
    mask = maskBlue + maskGreen + maskRed
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Blue Tracking Conts

    maskOpenBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_OPEN, kernelOpen)
    maskCloseBlue = cv2.morphologyEx(maskOpenBlue, cv2.MORPH_CLOSE, kernelClose)

    maskFinalBlue = maskCloseBlue
    _, contsBlue, h = cv2.findContours(maskFinalBlue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(res, contsBlue, -1, (255, 0, 0), 3)
    for i in range(len(contsBlue)):
        x, y, w, h = cv2.boundingRect(contsBlue[i])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, textBlue, (x, y + h), font, 1.0, (0, 255, 255))

    #Red Tracking Conts

    maskOpenRed = cv2.morphologyEx(maskRed, cv2.MORPH_OPEN, kernelOpen)
    maskCloseRed = cv2.morphologyEx(maskOpenRed, cv2.MORPH_CLOSE, kernelClose)

    maskFinalRed = maskCloseRed
    _, contsRed, h = cv2.findContours(maskFinalRed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(res, contsRed, -1, (255, 0, 0), 3)
    for i in range(len(contsRed)):
        x, y, w, h = cv2.boundingRect(contsRed[i])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, textRed, (x, y + h), font, 1.0, (0, 255, 255))

     # Green Tracking Conts

    maskOpenGreen = cv2.morphologyEx(maskGreen, cv2.MORPH_OPEN, kernelOpen)
    maskCloseGreen = cv2.morphologyEx(maskOpenGreen, cv2.MORPH_CLOSE, kernelClose)

    maskFinalGreen = maskCloseGreen
    _, contsGreen, h = cv2.findContours(maskFinalGreen.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(res, contsGreen, -1, (255, 0, 0), 3)
    for i in range(len(contsGreen)):
        x, y, w, h = cv2.boundingRect(contsGreen[i])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, textGreen, (x, y + h), font, 1.0, (0, 255, 255))

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    # cv2.imshow('maskREd', maskRed)
    # cv2.imshow('maskGreen', maskGreen)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
