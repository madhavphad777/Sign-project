import os
import random
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading

# Constants for image and video display size
DISPLAY_WIDTH = 400
DISPLAY_HEIGHT = 300

def predict_word(text):
    return text

def fetch_image(word, image_dir='./images'):
    word_dir = os.path.join(image_dir, word)
    if not os.path.exists(word_dir):
        raise FileNotFoundError(f"No directory found for the word: {word}")

    images = [f for f in os.listdir(word_dir) if os.path.isfile(os.path.join(word_dir, f))]
    if not images:
        raise FileNotFoundError(f"No images found in directory: {word_dir}")
    
    image_path = os.path.join(word_dir, random.choice(images))
    return cv2.imread(image_path)

def fetch_video(word, video_dir='./videos'):
    word_dir = os.path.join(video_dir, word)
    if not os.path.exists(word_dir):
        raise FileNotFoundError(f"No directory found for the word: {word}")

    videos = [f for f in os.listdir(word_dir) if os.path.isfile(os.path.join(word_dir, f))]
    if not videos:
        raise FileNotFoundError(f"No videos found in directory: {word_dir}")

    video_path = os.path.join(word_dir, random.choice(videos))
    return video_path

def display_image(image, label):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    label.config(image=image)
    label.image = image

def play_video(video_path, label):
    cap = cv2.VideoCapture(video_path)

    def update_frame():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if no frame is returned

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            label.config(image=frame)
            label.image = frame

            # Pause for a brief moment to create a video effect
            cv2.waitKey(33)

        cap.release()

    threading.Thread(target=update_frame).start()

def on_predict():
    text = input_entry.get()
    word = predict_word(text)
    
    try:
        image = fetch_image(word)
        display_image(image, image_label)
    except FileNotFoundError as e:
        image_label.config(text=str(e))
    
    try:
        global video_path
        video_path = fetch_video(word)
        play_video(video_path, video_label)
    except FileNotFoundError as e:
        video_label.config(text=str(e))

def on_play_again():
    if video_path:
        play_video(video_path, video_label)

def close_app():
    root.destroy()

root = tk.Tk()
root.title("Image and Video Display")

input_entry = tk.Entry(root, width=50)
input_entry.pack()

predict_button = tk.Button(root, text="Predict", command=on_predict)
predict_button.pack()

image_label = tk.Label(root, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
image_label.pack()

video_label = tk.Label(root, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
video_label.pack()

play_again_button = tk.Button(root, text="Replay", command=on_play_again)
play_again_button.pack()

close_button = tk.Button(root, text="Close", command=close_app)
close_button.pack()

video_path = None

root.mainloop()
