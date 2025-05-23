import torch
from model import DoodleCNN
import torch.nn as nn
import cv2
import numpy as np

class DoodleInferencer:
    def __init__(self, model_path): 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.saved_data = torch.load(model_path, weights_only = False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path.")

        self.classes = self.saved_data["classes"]
        self.image_size = self.saved_data["image_size"]
    
    def __loadModel(self):
        self.model = DoodleCNN(input_size = self.image_size, num_classes = len(self.classes))
        self.model.load_state_dict(self.saved_data["model"])
        self.model.to(self.device)
        self.model.eval()
    
    def __preprocessImage(self, image):
        #Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = 255 - image     # Invert: black bg, white fg

        #find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise ValueError("No object found in image.")
    
        #Get largest contour
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cropped = image[y:y+h, x:x+w]

        # Make the bounding box is 80% of the self.image_size, others are padding
        target_size = int(self.image_size * 0.8)
        if h > w:
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            new_w = target_size
            new_h = int(h * (target_size / w))
        
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Add padding to make image of size self.image_size x self.image_size
        pad_top = (self.image_size - new_h) // 2
        pad_bottom = self.image_size - new_h - pad_top
        pad_left = (self.image_size - new_w) // 2
        pad_right = self.image_size - new_w - pad_left
        padded = np.pad( resized, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

        # Denoise and threshold
        median = cv2.medianBlur(padded, 1)
        gaussian = cv2.GaussianBlur(median, (3, 3), 0)
        _, image_resized = cv2.threshold(gaussian, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Normalize and convert to tensor
        image_resized = image_resized.astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0)

        return tensor_image

    def getClasses(self):
        return self.saved_data["classes"]
    
    def inference(self, image):
        self.__loadModel()
        tensor_image = self.__preprocessImage(image)
        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            tensor_image = tensor_image.to(self.device)
            output = self.model(tensor_image)
            probs = softmax(output)
            
            predicted_class_id = torch.argmax(probs, dim=1).item()
            predicted_class = self.classes[predicted_class_id]
            predicted_class_prob = probs[0][predicted_class_id].item()

            return predicted_class, predicted_class_prob

if __name__ == "__main__":
    # Example usage
    image_path = "./doodle_classification/test_img/envelope.png"  
    image = cv2.imread(image_path)
    inferencer = DoodleInferencer(model_path = "./doodle_classification/checkpoint/best.pt")
    predicted_class, predicted_class_prob = inferencer.inference(image)
    print(f"Predicted class: {predicted_class}, Probability: {predicted_class_prob:.4f}")

