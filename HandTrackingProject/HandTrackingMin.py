import cv2
import mediapipe as mp
import time

# Create video object
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

previous_Time = 0
current_Time = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)  When something detected or not?(If you show your hand to the camera, the information will come, otherwise None)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Check index number
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                # Get pixels
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                if id == 4:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    current_Time = time.time()
    fps = 1 / (current_Time - previous_Time)
    previous_Time = current_Time

    cv2.putText(
        img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
    )

    cv2.imshow("Image", img)
    cv2.waitKey(1)
