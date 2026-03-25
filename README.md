
# Real-Time Emotion Recognition 🎭

A computer vision project that detects and classifies human emotions in real-time using a webcam. The system uses Haar Cascades for face detection and a deep learning model for classification.

## 🚀 Evolution & Performance
The project evolved through three main stages of development:
1. **Custom CNN (Initial):** Trained on a small custom dataset, achieving ~**55%** accuracy.
2. **Custom CNN (Expanded):** Switched to a larger 3-class dataset (Happy, Sad, Neutral), boosting accuracy to ~**70%**.
3. **ResNet18 (Transfer Learning):** The final version utilizes a modified **ResNet18** architecture. By adapting the input layer for grayscale and fine-tuning, the accuracy reached **75-80%**.

## 🛠 Features
* **Real-time processing:** Optimized frame handling for smooth performance.
* **Prediction Smoothing:** Uses a voting mechanism over a sliding window of frames to eliminate flickering.
* **Visual Overlay:** Displays real-time confidence scores and corresponding emojis.

## 📁 Project Structure
* `webcam_demo.py`: Main script for live detection.
* `train.py`: Training script including data augmentation and class balancing.
* `weights/`: Folder containing the pre-trained weights (`.pth`).
* `assets/`: Emoji images for the UI.

## 📊 Dataset
To train the model, you need to provide your own dataset. This project is optimized for a 3-class classification (Happy, Sad, Neutral).

### Recommended Datasets:
* [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) (Highly recommended)
* [AffectNet](http://mohammadmahoor.com/affectnet/)

### Data Preparation:
1. Create a `dataset/data/` folder in the root directory.
2. Organize your images into `train` and `test` subfolders as follows:
   ```text
   data/
   ├── train/
   │   ├── happy/
   │   ├── sad/
   │   └── neutral/
   └── test/
       ├── happy/
       ├── sad/
       └── neutral/
3. You can add new emotion classes, but this will require modifying the model's structure for training 
   
## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Nazar4s/CheckEmoAI.git](https://github.com/your-username/emotion-recognition.git)
   cd emotion-recognition
