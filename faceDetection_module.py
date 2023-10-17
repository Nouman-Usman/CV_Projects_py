import cv2
import mediapipe as mp
import time


class faceDetection():
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.mpPose = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpPose.FaceDetection(self.min_detection_confidence)

    def findFace(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # if self.results.detections:
        #     if draw:
        #         self.mpDraw.draw_detection(img, self.results.detections)
        return img

    def getPoint(self, img, poseNo=0, draw=True):
        lmList = []
        if self.results.detections:
            for id, lm in enumerate(self.results.detections):
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
                lmList.append([id, box])
                if draw:
                    img = self.fancyDraw(img, box)
                    cv2.putText(img, f"{(int(lm.score[0] * 100))}", (boxX, boxY - 20), cv2.FONT_HERSHEY_PLAIN, 1,
                                (122, 0, 222), 2)
        return lmList,img

    def fancyDraw(self, img, bbox, l=30, t= 7):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        # Left top corner
        cv2.line(img, (x, y), (x, y+l), (112, 22, 223), t)
        cv2.line(img, (x, y), (x + l, y), (12, 22, 223), t)
        # Right bottom corner
        cv2.line(img, (x1, y1), (x1, y1-l), (112, 22, 223), t)
        cv2.line(img, (x1, y1), (x1 - l, y1), (12, 22, 223), t)
        # Right Top Corner
        cv2.line(img, (x1, y), (x1, y+l), (112, 22, 223), t)
        cv2.line(img, (x1, y), (x1 - l, y), (12, 22, 223), t)
        # Right Bottom Corner
        cv2.line(img, (x, y1), (x, y1-l), (112, 22, 223), t)
        cv2.line(img, (x, y1), (x + l, y1), (12, 22, 223), t)
        return img

def main():
    ctime = 0
    ptime = 0
    cap = cv2.VideoCapture("Videos/2.mp4")
    obj = faceDetection()
    while True:
        success, img = cap.read()
        img = obj.findFace(img)
        list,img = obj.getPoint(img)
        if len(list) != 0:
            print(list[0])
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_ITALIC, 3, (12, 0, 22), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(5)


if __name__ == "__main__":
    main()
