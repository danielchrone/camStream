import cv2
import numpy as np
import math

kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))
cap = cv2.VideoCapture("test2_crop2.mp4")
font = cv2.FONT_HERSHEY_SIMPLEX

textBlue = "Shoulder"
textRed = "Elbow"
textGreen = "Hand"

def distance(x1_y1,x2_y2):
    x1,y1 =x1_y1
    x2,y2 =x2_y2
    dist = math.sqrt((math.fabs(x2-x1))**2+((math.fabs(y2-y1)))**2)
    return dist


while 1:
    _, frame = cap.read()
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([0, 168, 0])
    upper_blue = np.array([105, 255, 204])

    lower_green = np.array([60, 40, 0])
    upper_green = np.array([83, 152, 255])

    lower_red = np.array([123, 134, 0])
    upper_red = np.array([255, 255, 255])

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
        blueRectWidth = int(w/2)
        blueRectHeight = int(h/2)
        blueXCenter = int(x+blueRectWidth)
        blueYCenter = int(y+blueRectHeight)


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
        redRectWidth = int(w/2)
        redRectHeight = int(h/2)
        redXCenter = int(x+redRectWidth)
        redYCenter = int(y+redRectHeight)

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
        greenRectWidth = int(w/2)
        greenRectHeight = int(h/2)
        greenXCenter = int(x+greenRectWidth)
        greenYCenter = int(y+greenRectHeight)

    cv2.line(frame, (redXCenter,redYCenter), (blueXCenter,blueYCenter),(255,255,255),2)
    cv2.line(frame, (redXCenter, redYCenter), (greenXCenter, greenYCenter), (255, 255, 255), 2)

    # Albue/Skulder trekant
    hypotenuse = distance((redXCenter,redYCenter), (blueXCenter,blueYCenter))
    horizontal = distance((redXCenter,redYCenter),(blueXCenter,redYCenter))
    thirdline = distance((blueXCenter,blueYCenter), (blueXCenter,redYCenter))
    angle = np.arcsin((thirdline/hypotenuse))*180/math.pi
    cv2.line(frame,(redXCenter,redYCenter),(blueXCenter,blueYCenter),(255,255,255),2)
    cv2.line(frame, (redXCenter, redYCenter), (blueXCenter, redYCenter), (255, 255, 255), 2)
    cv2.line(frame, (blueXCenter, blueYCenter), (blueXCenter, redYCenter), (255, 255, 255), 2)
    cv2.putText(frame,str(int(angle)),(redXCenter-30,redYCenter),font, 1.0, (0, 255, 255))
    
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    # cv2.imshow('maskREd', maskRed)
    # cv2.imshow('maskGreen', maskGreen)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
