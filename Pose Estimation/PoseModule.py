import cv2
import mediapipe as mp
import time


class PoseDetector:
    def __init__(
        self,
        mode=False,
        complexity=1,
        smooth=True,
        segmentation=False,
        smooth_segmentation=True,
        detectionCon=0.5,
        trackCon=0.5,
    ):

        self.mode = mode
        self.complexity = complexity
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode,
            self.complexity,
            self.smooth,
            self.segmentation,
            self.smooth_segmentation,
            self.detectionCon,
            self.trackCon,
        )

    def find_pose(self, img, draw=True):

        # Convert your img Mediapipe user RGB Format
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Send this img to our module
        self.results = self.pose.process(imgRGB)

        # print(results.pose_landmarks)
        # Draw Landmarks
        if self.results.pose_landmarks:
            if draw:
                # Connect lines (mpPose.POSECONNECTIONS)
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
                )
        return img

    def getPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                # Get pixels
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture("PoseVideos/jim-carrey-i-ve-got-the-power.mp4")
    # cap = cv2.VideoCapture(
    #     "PoseVideos/tennis-forehand-slow-motion-simon-top-tennis.mp4"
    # )
    # cap = cv2.VideoCapture("PoseVideos/Top 10 Dunks of The Decade.mp4")
    previous_time = 0
    # Create object from PoseDetector Class
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.find_pose(img=img)

        lmList = detector.getPosition(img=img)
        if len(lmList) != 0:
            print(lmList[14])
            # draw spesific point if u want
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 10, (0, 0, 255), cv2.FILLED)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(
            img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
