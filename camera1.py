# import os
# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np

# # Directories and Parameters
# DATA_DIR = './data'
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

# number_of_classes = 3
# dataset_size = 100

# # Load the trained model
# model_dict = pickle.load(open('./model2hand.p', 'rb'))
# model = model_dict['model']

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.5)

# # Labels dictionary for prediction
# labels_dict = {0: 'Bhupesh', 1: 'Madhav', 2: 'Sai'}

# # Initialize video capture
# camera_indices = [0, 1, 2]  # List of camera indices to try

# for index in camera_indices:
#     cap = cv2.VideoCapture(index)
#     if cap.isOpened():
#         print(f"Camera opened successfully with index {index}")
#         break
# else:
#     print("Error: Could not open video stream or file")
#     exit()

# paused = False
# predicted_character = ""

# while True:
#     if not paused:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture image")
#             break

#         # Prediction logic
#         data_aux = []
#         x_ = []
#         y_ = []

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style()
#                 )

#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     x_.append(x)
#                     y_.append(y)

#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))
#                     data_aux.append(y - min(y_))

#                 x1 = int(min(x_) * frame.shape[1]) - 10
#                 y1 = int(min(y_) * frame.shape[0]) - 10
#                 x2 = int(max(x_) * frame.shape[1]) + 10
#                 y2 = int(max(y_) * frame.shape[0]) + 10

#                 # Ensure data_aux has the correct number of features
#                 expected_features = 42
#                 if len(data_aux) > expected_features:
#                     data_aux = data_aux[:expected_features]
#                 elif len(data_aux) < expected_features:
#                     data_aux.extend([0] * (expected_features - len(data_aux)))

#                 # Make a prediction
#                 prediction = model.predict([np.asarray(data_aux)])
#                 predicted_character = prediction[0]  # Use the prediction directly if it is a string

#                 # Draw rectangle around hand and text for the prediction
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
#                 cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

#         # Display the predicted character in the top-left corner
#         cv2.putText(frame, f'Predicted: {predicted_character}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#     cv2.imshow('frame', frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):  # Toggle pause/play with 'q'
#         paused = not paused
#     elif key == ord('s'):  # Exit with 's'
#         break

# cap.release()
# cv2.destroyAllWindows()










# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ two hand camea recognition code  +++++++++++++++++++++++++++++
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ two hand camea recognition code  +++++++++++++++++++++++++++++
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ two hand camea recognition code  +++++++++++++++++++++++++++++
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ two hand camea recognition code  +++++++++++++++++++++++++++++


# # import os
# # import pickle
# # import cv2
# # import mediapipe as mp
# # import numpy as np

# # # Directories and Parameters
# # DATA_DIR = './data'
# # if not os.path.exists(DATA_DIR):
# #     os.makedirs(DATA_DIR)

# # number_of_classes = 3
# # dataset_size = 100

# # # Load the trained model
# # model_dict = pickle.load(open('./model2.p', 'rb'))
# # model = model_dict['model']

# # # Initialize MediaPipe Hands
# # mp_hands = mp.solutions.hands
# # mp_drawing = mp.solutions.drawing_utils
# # mp_drawing_styles = mp.solutions.drawing_styles
# # hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.5)

# # # Labels dictionary for prediction
# # labels_dict = {0: 'Bhupesh', 1: 'Madhav', 2: 'Sai'}

# # # Initialize video capture
# # camera_indices = [0, 1, 2]  # List of camera indices to try

# # for index in camera_indices:
# #     cap = cv2.VideoCapture(index)
# #     if cap.isOpened():
# #         print(f"Camera opened successfully with index {index}")
# #         break
# # else:
# #     print("Error: Could not open video stream or file")
# #     exit()

# # paused = False
# # predicted_characters = ["", ""]

# # while True:
# #     if not paused:
# #         ret, frame = cap.read()
# #         if not ret:
# #             print("Error: Failed to capture image")
# #             break

# #         # Prediction logic
# #         data_aux = [[], []]  # Two hands, two lists
# #         x_ = [[], []]
# #         y_ = [[], []]

# #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         results = hands.process(frame_rgb)

# #         if results.multi_hand_landmarks:
# #             for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
# #                 if hand_index >= 2:  # Limit to two hands
# #                     break

# #                 # Draw hand landmarks
# #                 mp_drawing.draw_landmarks(
# #                     frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
# #                     mp_drawing_styles.get_default_hand_landmarks_style(),
# #                     mp_drawing_styles.get_default_hand_connections_style()
# #                 )

# #                 for i in range(len(hand_landmarks.landmark)):
# #                     x = hand_landmarks.landmark[i].x
# #                     y = hand_landmarks.landmark[i].y
# #                     x_[hand_index].append(x)
# #                     y_[hand_index].append(y)

# #                 for i in range(len(hand_landmarks.landmark)):
# #                     x = hand_landmarks.landmark[i].x
# #                     y = hand_landmarks.landmark[i].y
# #                     data_aux[hand_index].append(x - min(x_[hand_index]))
# #                     data_aux[hand_index].append(y - min(y_[hand_index]))

# #                 x1 = int(min(x_[hand_index]) * frame.shape[1]) - 10
# #                 y1 = int(min(y_[hand_index]) * frame.shape[0]) - 10
# #                 x2 = int(max(x_[hand_index]) * frame.shape[1]) + 10
# #                 y2 = int(max(y_[hand_index]) * frame.shape[0]) + 10

# #                 # Ensure data_aux has the correct number of features
# #                 expected_features = 42
# #                 if len(data_aux[hand_index]) > expected_features:
# #                     data_aux[hand_index] = data_aux[hand_index][:expected_features]
# #                 elif len(data_aux[hand_index]) < expected_features:
# #                     data_aux[hand_index].extend([0] * (expected_features - len(data_aux[hand_index])))

# #                 # Make a prediction
# #                 prediction = model.predict([np.asarray(data_aux[hand_index])])
# #                 predicted_characters[hand_index] = prediction[0]  # Use the prediction directly if it is a string

# #                 # Draw rectangle around hand and text for the prediction
# #                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
# #                 cv2.putText(frame, predicted_characters[hand_index], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

# #         # Display the predicted characters
# #         cv2.putText(frame, f'Predicted 1: {predicted_characters[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
# #         cv2.putText(frame, f'Predicted 2: {predicted_characters[1]}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# #     cv2.imshow('frame', frame)

# #     key = cv2.waitKey(1) & 0xFF
# #     if key == ord('q'):  # Toggle pause/play with 'q'
# #         paused = not paused
# #     elif key == ord('s'):  # Exit with 's'
# #         break

# # cap.release()
# # cv2.destroyAllWindows()



import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Directories and Parameters
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Load the trained model
model_dict = pickle.load(open('model3hand.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.5)

# Labels dictionary for prediction
labels_dict = {0: 'Bhupesh', 1: 'Madhav', 2: 'Sai'}

# Initialize video capture
camera_indices = [0, 1, 2]  # List of camera indices to try

for index in camera_indices:
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera opened successfully with index {index}")
        break
else:
    print("Error: Could not open video stream or file")
    exit()

paused = False
predicted_character = ""

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Prediction logic
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
            x1 = int(min(x_) * frame.shape[1]) - 10
            y1 = int(min(y_) * frame.shape[0]) - 10
            x2 = int(max(x_) * frame.shape[1]) + 10
            y2 = int(max(y_) * frame.shape[0]) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

        # Display the predicted character in the top-left corner
        cv2.putText(frame, f'Predicted: {predicted_character}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Toggle pause/play with 'q'
        paused = not paused
    elif key == ord('s'):  # Exit with 's'
        break

cap.release()
cv2.destroyAllWindows()
