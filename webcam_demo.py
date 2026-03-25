import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import cv2
import collections
import os

def load_model(weights_path, device):
    """Initialize and load ResNet18 weights."""
    # Using ResNet18 as the core architecture
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Adaptation for single-channel (Grayscale) input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    setattr(model, 'maxpool', nn.Identity())
    
    num_ftrs = model.fc.in_features
    # 3 output classes: Happy, Sad, Neutral
    model.fc = nn.Linear(num_ftrs, 3) 
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at: {weights_path}")
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # Configuration
    weights_path = './weights/Em0_00.pth'
    
    try:
        model = load_model(weights_path, device).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Image preprocessing pipeline
    trms = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return
    
    print("Webcam started. Press 'q' to quit.")

    # Dictionary mapping class indices to emoji images
    emoji_dict = {
        0: cv2.imread('assets/smile_china.jpg'),
        2: cv2.imread('assets/sad_china.jpg'),
        1: cv2.imread('assets/n_china.jpg')
    }

    # smoothing: stores the last 5 predictions to prevent flickering
    predictions_history = collections.deque(maxlen=5)
    last_stable_index = 2  # Default to "Neutral"
    last_conf = 0.0
    frame_to_skip = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Process every 6th frame to improve performance
        if frame_to_skip > 5: 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Select the largest face detected
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Preprocess face for the model
                face_img = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_img, (112, 112))
                
                img_t = trms(Image.fromarray(face_resized))
                batch_t = img_t.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(batch_t)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    conf, index = torch.max(probs, 1)
                    
                    predictions_history.append(index.item())
                    last_conf = conf.item()

                # Voting: pick the most frequent prediction from history
                if predictions_history:
                    last_stable_index = max(set(predictions_history), key=predictions_history.count)
            
            frame_to_skip = 0
        else:
            frame_to_skip += 1

        # UI Overlay: Draw emoji and confidence text
        emoji = emoji_dict.get(last_stable_index)
        if emoji is not None:
            emoji_res = cv2.resize(emoji, (120, 120))
            # Overlay emoji in the top-left corner
            frame[10:130, 10:130] = emoji_res
            
            text = f"Emotion ID: {last_stable_index} (Conf: {last_conf:.2f})"
            cv2.putText(frame, text, (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Emotion AI Detector", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
