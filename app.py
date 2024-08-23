from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from flask_cors import CORS
import base64
from PIL import Image, ImageEnhance
from io import BytesIO
import numpy as np
import pickle
from email.message import EmailMessage
import ssl
import smtplib
import os
import random
from PIL import Image
import cv2
import mediapipe as mp
from flask_cors import CORS
import subprocess 
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')

DEFAULT_SENDER_EMAIL = 'madhavphad4321@gmail.com'
DEFAULT_SENDER_PASSWORD = 'uqgp uxux ejmw gnma'


app = Flask(__name__)
CORS(app)  # Enable CORS
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load your model
model_dict = pickle.load(open('model3hand.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


labels_dict = {0: 'A', 1: 'B', 2: 'L',3 : 'accept'}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/communication')
def services():
    return render_template('services.html')

# @app.route('/career')
# def career():
#     return render_template('https://job-portal-2-p5bv.onrender.com/')
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email_receiver = request.form['email']
        subject = request.form['subject']
        message = request.form['message']

        body = f"Name: {name}\nEmail: {email_receiver}\nMessage: {message}"

        try:
            send_email(DEFAULT_SENDER_EMAIL, DEFAULT_SENDER_PASSWORD, email_receiver, subject, body)
            flash('Message sent successfully!', 'success')
        except Exception as e:
            print(f"Error: {e}")
            flash('An error occurred while sending the message. Please try again later.', 'danger')

        return redirect(url_for('contact'))

    return render_template('contact.html')

def send_email(email_sender, email_password, email_receiver, subject, body):
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image']
        print(f"Received Image Data: {image_data[:50]}...")  # Print initial part of the base64 data

        # Remove the data URL prefix
        image_data = image_data.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))

        # Adjust brightness and contrast
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.5)  # Increase brightness
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)  # Increase contrast

        image = np.array(image)

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
                data_aux = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(x_[i] - min(x_))
                    data_aux.append(y_[i] - min(y_))

                # Ensure data_aux has the correct number of features
                expected_features = 42
                if len(data_aux) > expected_features:
                    data_aux = data_aux[:expected_features]
                elif len(data_aux) < expected_features:
                    data_aux.extend([0] * (expected_features - len(data_aux)))

                prediction = model.predict([np.asarray(data_aux)])
                try:
                    predicted_character = str(prediction[0])  # Use the prediction directly if it is a string
                    print(f"Prediction: {predicted_character}")  # Debug print
                    return jsonify({'prediction': predicted_character})
                except (ValueError, KeyError) as e:
                    print(f"Prediction Error: {e}")
                    return jsonify({'error': str(e)}), 500

        print("No hand detected")
        return jsonify({'prediction': 'No hand detected'})

    except Exception as e:
        print(f"Error: {e}")  # Debug print
        return jsonify({'error': str(e)}), 500



#   ++++++++++++++++++++++++++++ updload image or video based on this predict the word +++++++++++++++++++++++++++++++++++++
#   ++++++++++++++++++++++++++++ updload image or video based on this predict the word +++++++++++++++++++++++++++++++++++++
#   ++++++++++++++++++++++++++++ updload image or video based on this predict the word +++++++++++++++++++++++++++++++++++++


DISPLAY_WIDTH = 400
DISPLAY_HEIGHT = 300

def predict_word(text):
    return text

def fetch_image(word, image_dir='./static/images'):
    word_dir = os.path.join(image_dir, word)
    if not os.path.exists(word_dir):
        raise FileNotFoundError(f"No directory found for the word: {word}")

    images = [f for f in os.listdir(word_dir) if os.path.isfile(os.path.join(word_dir, f))]
    if not images:
        raise FileNotFoundError(f"No images found in directory: {word_dir}")
    
    image_path = os.path.join(word_dir, random.choice(images))
    return image_path

def fetch_video(word, video_dir='./static/videos'):
    word_dir = os.path.join(video_dir, word)
    if not os.path.exists(word_dir):
        raise FileNotFoundError(f"No directory found for the word: {word}")

    videos = [f for f in os.listdir(word_dir) if os.path.isfile(os.path.join(word_dir, f))]
    if not videos:
        raise FileNotFoundError(f"No videos found in directory: {word_dir}")

    video_path = os.path.join(word_dir, random.choice(videos))
    return video_path


@app.route('/open_camera', methods=['POST'])
def open_camera():
    try:
        # Use subprocess to run camera.py using bash
        subprocess.Popen(['python', 'camera.py'])
        return jsonify({'message': 'Camera opened successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500@app.route('/open_camera', methods=['POST'])
def open_camera():
    try:
        # Use subprocess to run camera.py using bash
        subprocess.Popen(['python', 'camera.py'])
        return jsonify({'message': 'Camera opened successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/word', methods=['POST'])
def predict_output():
    text = request.form['input_text']
    word = predict_word(text)
    response = {'status': 'ok'}
    
    try:
        image_path = fetch_image(word)
        response['image_path'] = image_path
    except FileNotFoundError as e:
        response['image_error'] = str(e)
    
    try:
        video_path = fetch_video(word)
        response['video_path'] = video_path
    except FileNotFoundError as e:
        response['video_error'] = str(e)
    
    return response

#   ++++++++++++++++++++++++++++ updload image or video based on this predict the word +++++++++++++++++++++++++++++++++++++
#   ++++++++++++++++++++++++++++ updload image or video based on this predict the word +++++++++++++++++++++++++++++++++++++
#   ++++++++++++++++++++++++++++ updload image or video based on this predict the word +++++++++++++++++++++++++++++++++++++


def process_frame(frame):
    data_aux = []
    x_ = []
    y_ = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = ""
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
            try:
                predicted_character = labels_dict[prediction[0]]
            except KeyError:
                predicted_character = prediction[0]  # Directly use the predicted value if it's not in labels_dict

    return frame, predicted_character

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('process_file', filename=file.filename))

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/process/<filename>')
def process_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext in ['.jpg', '.jpeg', '.png']:
        frame = cv2.imread(filepath)
        frame, predicted_character = process_frame(frame)
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
        cv2.imwrite(processed_filepath, frame)
        return render_template('result.html', image_url=url_for('send_file', filename='processed_' + filename), prediction=predicted_character)

    elif file_ext in ['.mp4', '.avi', '.mov']:
        cap = cv2.VideoCapture(filepath)
        output_filename = 'processed_' + filename
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filepath, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

        predictions = []
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame, predicted_character = process_frame(frame)
                predictions.append(predicted_character)
                out.write(frame)
        except Exception as e:
            print(f"Error processing video: {e}")
        finally:
            cap.release()
            out.release()

        if os.path.exists(output_filepath):
            most_common_prediction = max(set(predictions), key=predictions.count)
            video_url = url_for('send_file', filename=output_filename, _external=True)
            print(f"Generated video URL: {video_url}")
            return render_template('result.html', video_url=video_url, prediction=most_common_prediction)
        else:
            return "Error processing video."

    else:
        return "Unsupported file format"





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
