import os
import time

import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils
fps = 0
current_Time = 0
previous_Time = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    if result.multi_hand_landmarks:
        for lm in result.multi_hand_landmarks:
            for id, Lm in enumerate(lm.landmark):
                print(id, Lm)
                height, width, c = img.shape
                cx, cy = int((Lm.x * width)), int((Lm.y * height))
                print(cx, cy)
            mpDraw.draw_landmarks(img, lm, mphands.HAND_CONNECTIONS)
    current_Time = time.time()
    fps = 1 / (current_Time - previous_Time)
    previous_Time = current_Time
    cv2.putText(
        img, str(int(fps)), (10, 70), cv2.Formatter_FMT_PYTHON, 1.6, (155, 0, 233), 2
    )
    cv2.imshow("Image", img)
    cv2.waitKey(1)
