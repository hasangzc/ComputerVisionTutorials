import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
# cap = cv2.VideoCapture("PoseVideos/jim-carrey-i-ve-got-the-power.mp4")
cap = cv2.VideoCapture("PoseVideos/tennis-forehand-slow-motion-simon-top-tennis.mp4")
# cap = cv2.VideoCapture("PoseVideos/Top 10 Dunks of The Decade.mp4")


while True:
    success, img = cap.read()

    # Convert your img Mediapipe user RGB Format
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Send this img to our module
    results = pose.process(imgRGB)

    # print(results.pose_landmarks)
    # Draw Landmarks
    if results.pose_landmarks:
        # Connect lines (mpPose.POSECONNECTIONS)
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        # Get landmarks
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            # print(id, lm)
            # Get pixels
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

    # Frame rate so high.
    # CHECK FPS
    current_time = time.time()
    previous_time = 0
    fps = 1 / (current_time - previous_time)

    cv2.putText(
        img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
    )
    cv2.imshow("Image", img)
    cv2.waitKey(1)
