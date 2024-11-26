import cv2
from load import FacialExpressionModel
import numpy as np
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()  # Read a frame from the webcam
        if fr is None:
            print("Error: Failed to capture frame")
            return None
        
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)  # Detect faces in the image
        
        # Check if any faces were detected
        if len(faces) == 0:
            print("No faces detected.")
        
        for (x, y, w, h) in faces:
            # Extract the face region from the image
            fc = gray_fr[y:y+h, x:x+w]
            
            # Check if the region is non-empty
            if fc.size == 0:
                print("Error: Empty face region.")
                continue
            
            print(f"Face region shape: {fc.shape}")  # Debugging: Check the shape of the face region
            
            # Resize the face region to 48x48
            roi = cv2.resize(fc, (48, 48))
            
            # Call the predict function to predict the emotion
            pred = model.predict_emotion(roi)  # Pass the resized ROI to the model
            
            # Display the predicted emotion on the frame
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        return fr

def gen(camera):
    while True:
        frame = camera.get_frame()
        cv2.imshow('Facial Expression Recognization',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
gen(VideoCamera())