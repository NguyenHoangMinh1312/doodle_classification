"""Do the inference of the hand drawing classification model"""
from argparse import ArgumentParser
import torch
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, resnet18
import torch.nn as nn
import cv2
import torchvision.transforms as T
import numpy as np


def get_args():
    parser = ArgumentParser(description = "Hand drawing classification")

    parser.add_argument("--model_path", "-m", type = str, default = "./doodle_classification/checkpoint/best.pt", help = "path to the model")
    parser.add_argument("--image_path", "-i", type = str, default = "./doodle_classification/test_img/sun.png", help = "path to the image")
    parser.add_argument("--threshold", "-t", type = int, default = 128, help = "threshold value for binary conversion")
    parser.add_argument("--verbose", "-v", action="store_true", help = "show detailed prediction scores")

    args = parser.parse_args()
    return args

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_data = torch.load(args.model_path, weights_only=False)
    classes = saved_data["classes"]
    
    # Set up model (in evaluation mode)
    model = resnet18(weights=None)    
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.to(device)
    model.load_state_dict(saved_data["model"])
    model.to(device)
    model.eval()  # Set evaluation mode before processing

    # Read original image for display
    display_image = cv2.imread(args.image_path)
    
    # Preprocess the image - matching training pipeline more closely
    image = cv2.imread(args.image_path).astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert to binary with white foreground on black background (if needed)
    # Checking if image needs inversion (assuming drawings are dark on light background)
    mean_value = np.mean(image)
    if mean_value > 128:  # Light background with dark drawing
        _, image = cv2.threshold(image, args.threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Resize after thresholding to preserve shape details
    image = cv2.resize(image, (224, 224))

    image /= 255.0
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # Add batch dimension

    # Inference
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        probs = softmax(output)
        
        # Get top prediction
        prediction = torch.argmax(probs, dim=1).item()
        prob = probs[0][prediction].item()
        
        # Display result
        result_text = f"{classes[prediction]}: {prob * 100:.2f}%"
        print(result_text)
        
        if args.verbose:
            # Print all class probabilities
            print("\nAll predictions:")
            sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
            for i, (p, idx) in enumerate(zip(sorted_probs[:5], sorted_indices[:5])):
                print(f"{classes[idx]}: {p.item() * 100:.2f}%")
        
        # Display image with prediction
        cv2.putText(display_image, result_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Prediction", display_image)
        cv2.waitKey(0)
    
if __name__ == "__main__":
    args = get_args()
    inference(args)