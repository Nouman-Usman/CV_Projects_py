import cv2
import mediapipe as mp
import time

ctime = 0
ptime = 0

cap = cv2.VideoCapture("Videos/2.mp4")
mpPose = mp.solutions.face_detection
faceDetection = mpPose.FaceDetection(0.80)
mpDraw = mp.solutions.drawing_utils
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    if results.detections:
        # mpDraw.draw_landmarks(img,results.detections)
        for id, lm in enumerate(results.detections):
            # print(lm.score)
            # mpDraw.draw_detection(img, lm)
            # print(lm.location_data.relative_bounding_box)
            bboxC = lm.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            boxX = int(bboxC.xmin * iw)
            boxY = int(bboxC.ymin * ih)
            boxH = int(bboxC.height * ih)
            boxW = int(bboxC.width * iw)
            box = boxX, boxY, boxW, boxH
            cv2.rectangle(img, box, (12, 0, 12), 3)
            cv2.putText(img,f"Score: {(int(lm.score[0] * 100))}" , (120, 450), cv2.FONT_HERSHEY_PLAIN, 3, (12, 0, 22), 3)
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_ITALIC, 3, (12, 0, 22), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(5)
