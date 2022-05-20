import os
import time

import cv2
import mediapipe as mp
import numpy as np

import HandTrackingModule as htm

folder_path = "img"
my_list = os.listdir(folder_path)
# print(my_list)
overlaylist = []
for imPath in my_list:
    image = cv2.imread(f"{folder_path}/{imPath}")
    overlaylist.append(image)
    # print(overlaylist)
# print(len(overlaylist))

header = overlaylist[0]
# Selected Color is Yellow
drawColor = (45, 255, 255)
# Determine Brush Thickness
brushThickness = 15
# Determine Eraser Thickness
eraserThickness = 200

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(detectionCon=0.85)
xp = 0
yp = 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)


while True:
    # 1.Import the image
    success, img = cap.read()
    # Flip for drawing. If i go right side its going to right and if i go left its going left
    img = cv2.flip(img, 1)

    # 2.Find Hand Landmarks
    img = detector.find_hands(img=img)
    lmList = detector.find_position(img, draw=False)
    if len(lmList) != 0:
        # print(lmList)
        # Index Fınger
        x1, y1 = lmList[8][1:]
        # Middle finger
        x2, y2 = lmList[12][1:]

        # 3.Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        # 4.If selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            # If we are in the header
            xp, yp = 0, 0
            if y1 < 125:
                # Yellow
                if 250 < x1 < 450:
                    header = overlaylist[0]
                    drawColor = (45, 255, 255)
                # Eraser
                elif 800 < x1 < 950:
                    header = overlaylist[1]
                    drawColor = (0, 0, 0)
                # Purple
                elif 100 < x1 < 200:
                    header = overlaylist[2]
                    drawColor = (255, 0, 255)
                # Green
                elif 500 < x1 < 750:
                    header = overlaylist[3]
                    drawColor = (152, 255, 152)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            print("Selection Mode!")
        # 5.If Drawing mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode!")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(
                    img,
                    (xp, yp),
                    (x1, y1),
                    drawColor,
                    eraserThickness,
                )
                cv2.line(
                    imgCanvas,
                    (xp, yp),
                    (x1, y1),
                    drawColor,
                    eraserThickness,
                )
            else:
                cv2.line(
                    img,
                    (xp, yp),
                    (x1, y1),
                    drawColor,
                    brushThickness,
                )
                cv2.line(
                    imgCanvas,
                    (xp, yp),
                    (x1, y1),
                    drawColor,
                    brushThickness,
                )
            xp, yp = x1, y1

    #
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    # When draw white-pannel black line
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    # Show my original colors!
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
