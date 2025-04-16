from flask import Flask, render_template, request, send_file
import cv2
import mediapipe as mp
import os
from pymongo import MongoClient
import gridfs
from io import BytesIO

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['pose_estimation']
fs = gridfs.GridFS(db)
app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}


def detect_and_draw_pose_webcam():
    cap = cv2.VideoCapture(0)  # Open webcam
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_and_draw_pose_in_video(video_path):
    video = cv2.VideoCapture(video_path)
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("bye")
            break

    video.release()
    cv2.destroyAllWindows()

import numpy as np  # Add this import for handling binary data

def detect_and_draw_pose_in_image(image_data, filename):
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                                  mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
    
    _, buffer = cv2.imencode('.jpg', image)
    output_filename = f"output_{filename}"
    fs.put(buffer.tobytes(), filename=output_filename)  # Store the processed image in MongoDB
    return output_filename

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/input')
def input():
    return render_template('inputpage.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/home')
def home():
    return render_template('home.html')

import tempfile

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the "Webcam" button was clicked
    if 'webcam' in request.form:
        detect_and_draw_pose_webcam()  # Open the webcam for pose detection
        return "Webcam pose detection completed. Close the webcam window to return."

    # Handle file upload
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    # Check if the file has a valid filename
    if file.filename == '':
        return "No selected file"
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file_data = file.read()  # Read the file as binary data
        fs.put(file_data, filename=filename)  # Store the file in MongoDB using GridFS
        
        if filename.endswith(('png', 'jpg', 'jpeg')):
            # Process the image and store the output in MongoDB
            output_filename = detect_and_draw_pose_in_image(file_data, filename)
            if output_filename:
                return render_template('result.html', image=output_filename)
        elif filename.endswith(('mp4', 'mkv')):
            # Use a temporary file for video processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(file_data)
                temp_video_path = temp_video.name
            
            detect_and_draw_pose_in_video(temp_video_path)  # Process the video
            
            # Optionally, store the processed video back in MongoDB
            processed_filename = f"processed_{filename}"
            with open(temp_video_path, 'rb') as processed_video:
                fs.put(processed_video.read(), filename=processed_filename)
            
            # Clean up the temporary file
            os.remove(temp_video_path)
            
            return render_template('result.html', video=processed_filename)

    return "Invalid file type"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_data = fs.find_one({'filename': filename})
    if not file_data:
        return "File not found", 404
    
    # Determine the mimetype based on the file extension
    if filename.endswith(('png', 'jpg', 'jpeg')):
        mimetype = 'image/jpeg'
    elif filename.endswith(('mp4', 'mkv')):
        mimetype = 'video/mp4'
    else:
        mimetype = 'application/octet-stream'
    
    return send_file(BytesIO(file_data.read()), mimetype=mimetype)

if __name__ == '__main__':
    app.run(debug=True)