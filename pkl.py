import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
warnings.filterwarnings('ignore', category=FutureWarning)  # Ignore FutureWarnings
warnings.filterwarnings('ignore', category=UserWarning)  # Ignore UserWarnings

# Your existing imports






import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
label_dict = {label: idx for idx, label in enumerate(os.listdir(DATA_DIR))}
for label, idx in label_dict.items():
    label_path = os.path.join(DATA_DIR, label)
    for img_path in os.listdir(label_path):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(label_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(idx)

# Save the data and label dictionary
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels, 'label_dict': label_dict}, f)
