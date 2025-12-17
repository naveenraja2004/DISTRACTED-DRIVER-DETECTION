from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import base64
import threading
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///drivers.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

db = SQLAlchemy(app)

# Load the trained model
model = load_model("driver_behavior_model.h5")
classes = ['Drowsy', 'Safe', 'Yawn', 'Face_tilt']
le = LabelEncoder()
le.fit(classes)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Database model
class Driver(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Create database tables
with app.app_context():
    db.create_all()

# Global variables for detection
detection_active = False
current_status = "Safe"
buzzer_active = False
buzzer_end_time = 0
detection_thread = None

# Detection function (similar to your detect.py)
def detection_loop():
    global detection_active, current_status, buzzer_active, buzzer_end_time

    # Start webcam
    cap = cv2.VideoCapture(0)
    
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        while detection_active:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape

                    # Get min/max coords to draw bounding box
                    x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
                    y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    # Draw bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Extract landmarks as feature vector
                    landmarks = []
                    for lm in face_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    landmarks = np.array(landmarks).reshape(1, -1)

                    # Predict class
                    pred = model.predict(landmarks)
                    class_id = np.argmax(pred)
                    current_status = le.inverse_transform([class_id])[0]

                    # Draw label above bounding box
                    cv2.putText(frame, current_status, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Activate buzzer for non-safe states
                    if current_status != 'Safe' and not buzzer_active:
                        buzzer_active = True
                        buzzer_end_time = time.time() + 5  # 5 seconds from now
            
            # Check if buzzer should be deactivated
            if buzzer_active and time.time() > buzzer_end_time:
                buzzer_active = False
            
            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Emit the frame and status to the client
            socketio.emit('detection_frame', {
                'image': 'data:image/jpeg;base64,' + frame_base64,
                'status': current_status,
                'buzzer_active': buzzer_active
            })
            
            # Control the frame rate
            time.sleep(0.03)  # Approximately 30 FPS
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        driver = Driver.query.filter_by(username=username).first()
        
        if driver and check_password_hash(driver.password, password):
            session['driver_id'] = driver.id
            session['username'] = driver.username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials', register=False)
    return render_template('login.html', register=False)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form.get('confirm-password')
        
        # Check if passwords match
        if password != confirm_password:
            return render_template('login.html', error='Passwords do not match', register=True)
        
        # Check if username already exists
        if Driver.query.filter_by(username=username).first():
            return render_template('login.html', error='Username already exists', register=True)
        
        # Create new driver
        hashed_password = generate_password_hash(password)
        new_driver = Driver(username=username, password=hashed_password)
        db.session.add(new_driver)
        db.session.commit()
        
        # Redirect to login page after successful registration
        return redirect(url_for('login'))
    
    return render_template('login.html', register=True)

@app.route('/dashboard')
def dashboard():
    if 'driver_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_active, detection_thread
    
    detection_active = not detection_active
    
    if detection_active:
        # Start detection thread
        detection_thread = threading.Thread(target=detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
    else:
        # Stop detection thread (will exit on next iteration)
        # We don't actually stop the thread, but it will exit when detection_active becomes False
        pass
    
    return jsonify({'status': 'success', 'active': detection_active})

@app.route('/get_status')
def get_status():
    global buzzer_active, buzzer_end_time
    
    # Check if buzzer should be deactivated
    if buzzer_active and time.time() > buzzer_end_time:
        buzzer_active = False
    
    return jsonify({'status': current_status, 'buzzer_active': buzzer_active})

if __name__ == '__main__':
    socketio.run(app, debug=True)