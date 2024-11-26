from tensorflow.keras.models import model_from_json
import cv2
import numpy as np
class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]
    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()
    def predict_emotion(self, img):
        # If the image has 3 channels (BGR), convert it to grayscale
        if len(img.shape) == 3 and img.shape[2] == 3:  # Check if image has 3 channels (BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if necessary
        
        # Resize image to 48x48 (if it's not already)
        img = cv2.resize(img, (48, 48))
        
        # Normalize the image (scaling pixel values to [0, 1])
        img = img.astype("float32") / 255  # Normalize pixel values between 0 and 1
        
        # Ensure the image has the correct shape for the model (48, 48, 1)
        img = np.expand_dims(img, axis=-1)  # Add channel dimension (grayscale)
        
        # Add batch dimension (for prediction) so the final shape is (1, 48, 48, 1)
        img = np.expand_dims(img, axis=0)   # Add batch dimension (shape becomes (1, 48, 48, 1))
        
        # Print the shape to verify
        print(f"Input shape for prediction: {img.shape}")
        
        # Predict the emotion using the model
        self.preds = self.loaded_model.predict(img)
        
        # Ensure the predictions have the correct shape
        print(f"Predictions: {self.preds}")
        print(f"Predictions shape: {self.preds.shape}")
        
        # Return the emotion corresponding to the highest prediction
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
