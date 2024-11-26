import cv2
from load import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            raise Exception("Webcam could not be opened. Please check the connection.")

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, fr = self.video.read()
        if fr is None:
            return None

        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        if len(faces) == 0:
            cv2.putText(fr, "No face detected", (50, 50), font, 1, (0, 255, 0), 2)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]
            if fc.size == 0:
                continue

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi)
            cv2.putText(fr, pred, (x, y - 10), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return fr

def gen(camera):
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                print("Failed to capture frame. Exiting...")
                break
            cv2.imshow('Facial Expression Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()

gen(VideoCamera())
