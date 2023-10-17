import time
import mediapipe as mp
import cv2


class hand_detetctor:
    def __init__(self, mode=False, maxHands=2, comp=1, minDetC=0.5, minTraC=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.comp = comp
        self.minDetC = minDetC
        self.minTraC = minTraC
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            self.mode, self.maxHands, self.comp, self.minDetC, self.minTraC
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        if self.result.multi_hand_landmarks:
            for lm in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, lm, self.mphands.HAND_CONNECTIONS)
        return img

    def findPos(self, img, handNO=0, draw=True):
        lmList = []
        if self.result.multi_hand_landmarks:
            my_hand = self.result.multi_hand_landmarks[handNO]
            for id, Lm in enumerate(my_hand.landmark):
                # print(id, Lm)
                height, width, c = img.shape
                cx, cy = int((Lm.x * width)), int((Lm.y * height))
                # print(cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (19, 0, 24), cv2.FILLED)

        return lmList


def main():
    detector = hand_detetctor()
    cap = cv2.VideoCapture(0)
    current_Time = 0
    fps = 0
    previous_Time = 0
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        list = detector.findPos(img)
        if len(list) != 0:
            print(list[4])
            print(list[20])
        current_Time = time.time()
        fps = 1 / (current_Time - previous_Time)
        previous_Time = current_Time
        cv2.putText(
            img, str(int(fps)), (10, 70), cv2.Formatter_FMT_PYTHON, 1.6, (15, 0, 233), 2
        )
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
