import cv2
import mediapipe as mp
import time


class poseDetetctor():
    def __init__(self, mode=False, modelC=1, smooth=True, enableSeg=False, smootheSeg=True, detC=0.5, tracC=0.5):
        self.mode = mode
        self.modelC = modelC
        self.smooth = smooth
        self.enableSeg = enableSeg
        self.smoothSeg = smootheSeg
        self.detC = detC
        self.tracC = tracC
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelC, self.smooth, self.enableSeg, self.smoothSeg, self.detC,
                                     self.tracC)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPoint(self,img,poseNo=0,draw = True):
        lmList = []
        if self.results.pose_landmarks:
            # my_hand = self.results.pose_landmarks.landmark[poseNo]
            for id, Lm in enumerate(self.results.pose_landmarks.landmark  ):
                # print(id, Lm)
                height, width, c = img.shape
                cx, cy = int((Lm.x * width)), int((Lm.y * height))
                # print(cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (19, 0, 24), cv2.FILLED)
        return lmList
        # for id, lm in enumerate(self.results.pose_landmarks.landmark):
            #     height, width, c = img.shape
            #     cx, cy = int((lm.x * width)), int((lm.y * height))
            #     print('cx', cx)
            #     print('cy', cy)


def main():
    ctime = 0
    ptime = 0
    obj = poseDetetctor()
    cap = cv2.VideoCapture("Videos/2.mp4")
    while True:
        success, img = cap.read()
        img = obj.findPose(img)
        list = obj.getPoint(img)
        print(list[4])
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_ITALIC, 3, (12, 0, 22), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
