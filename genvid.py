import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Directories and Parameters
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.5)

# Labels dictionary for prediction
labels_dict = {0: 'Bhupesh', 1: 'Madhav', 2: 'Sai'}

def process_frame(frame):
    data_aux = []
    x_ = []
    y_ = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

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

            x1 = int(min(x_) * frame.shape[1]) - 10
            y1 = int(min(y_) * frame.shape[0]) - 10
            x2 = int(max(x_) * frame.shape[1]) + 10
            y2 = int(max(y_) * frame.shape[0]) + 10

            # Ensure data_aux has the correct number of features
            expected_features = 42
            if len(data_aux) > expected_features:
                data_aux = data_aux[:expected_features]
            elif len(data_aux) < expected_features:
                data_aux.extend([0] * (expected_features - len(data_aux)))

            # Make a prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]  # Use the prediction directly if it is a string

            # Draw rectangle around hand and text for the prediction
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the predicted character in the top-left corner
    cv2.putText(frame, f'Predicted: {predicted_character}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    return frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_images(image_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            frame = cv2.imread(image_path)
            frame = process_frame(frame)
            cv2.imshow('frame', frame)
            cv2.waitKey(0)  # Wait for a key press to display the next image

    cv2.destroyAllWindows()

# Specify the input type (either 'image' or 'video') and the path
input_type = 'video'  # Change to 'image' for processing images
input_path = './images/able/00376_frame17.jpg'  # Change to directory path for images

if input_type == 'video':
    process_video(input_path)
elif input_type == 'image':
    process_images(input_path)
else:
    print("Invalid input type specified. Please use 'image' or 'video'.")
