import os
import cv2
import threading
import time
import queue
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from analyzer import InterviewAnalyzer
from datetime import datetime


from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///interview_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db = SQLAlchemy(app)


app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['VIDEO_FOLDER'] = 'static/videos'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEO_FOLDER'], exist_ok=True)


class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.String(20), nullable=False)
    emotion = db.Column(db.String(20), nullable=False)
    eye_contact = db.Column(db.String(20), nullable=False)
    posture = db.Column(db.String(20), nullable=False)
    wpm = db.Column(db.Integer, nullable=False)
    fillers = db.Column(db.Integer, nullable=False)
    


with app.app_context():
    db.create_all()


MODEL_FILE = "models/ultimate_holistic_model.keras" 
if not os.path.exists(MODEL_FILE):
    if os.path.exists("ultimate_holistic_model.keras"): 
        MODEL_FILE = "ultimate_holistic_model.keras"

print(f"--- Loading Engine with Model: {MODEL_FILE} ---")
engine = InterviewAnalyzer(model_path=MODEL_FILE)

class State:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.camera_on = True
        self.data = {
            "emotion": "Scanning...", 
            "confidence": "Analyzing...", 
            "eye_contact": "Tracking...", 
            "posture": "Checking...",
            "transcript": "Listening...", 
            "sentiment": "--", 
            "wpm": 0, 
            "fillers": 0
        }

state = State()
audio_q = queue.Queue()


def video_thread():
    cap = cv2.VideoCapture(0)
    while True:
        if not state.camera_on:
            time.sleep(0.5)
            continue
        ret, frame = cap.read()
        if not ret: 
            time.sleep(0.1)
            continue
        processed_frame, results = engine.analyze_visuals(frame)
        with state.lock:
            state.frame = processed_frame
            state.data.update(results)
        time.sleep(0.03)

def audio_listen_thread():
    if not engine.mic_active: return
    with engine.microphone as source:
        while True:
            if not state.camera_on: 
                time.sleep(1)
                continue
            try:
                audio = engine.recognizer.listen(source, timeout=2, phrase_time_limit=5)
                audio_q.put(audio)
            except: continue

def audio_process_thread():
    while True:
        try:
            audio = audio_q.get()
            txt, sent, fil, wpm = engine.process_audio(audio)
            if txt:
                with state.lock:
                    state.data["transcript"] = txt
                    state.data["sentiment"] = sent
                    state.data["fillers"] += fil
                    state.data["wpm"] = wpm
        except: pass



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/toggle_camera', methods=['POST'])
def toggle():
    """Start/Stop Camera and Save to DB"""
    state.camera_on = not state.camera_on
    
    
    if not state.camera_on:
        with state.lock:
            
            new_session = Session(
                timestamp = datetime.now().strftime("%Y-%m-%d %I:%M %p"),
                confidence = state.data['confidence'],
                emotion = state.data['emotion'],
                eye_contact = state.data.get('eye_contact', 'Unknown'),
                posture = state.data.get('posture', 'Unknown'),
                wpm = int(state.data['wpm']),
                fillers = int(state.data['fillers'])
            )
            
            try:
                db.session.add(new_session)
                db.session.commit()
                print("✅ Session Saved to Database!")
            except Exception as e:
                print(f"❌ DB Error: {e}")

    status = "ON" if state.camera_on else "OFF"
    return jsonify({"status": state.camera_on})

@app.route('/history')
def get_history():
    """Fetch history from Database instead of list"""
    
    sessions = Session.query.order_by(Session.id.desc()).limit(20).all()
    
    
    history_data = []
    for s in sessions:
        history_data.append({
            "timestamp": s.timestamp,
            "confidence": s.confidence,
            "emotion": s.emotion,
            "wpm": s.wpm,
            "fillers": s.fillers
        })
    
    return jsonify(history_data)

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            if not state.camera_on:
                blank = np.zeros((480, 640, 3), np.uint8)
                blank[:] = (30, 30, 30)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(blank, "Analysis Paused", (200, 240), font, 1, (255,255,255), 2)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', blank)[1].tobytes() + b'\r\n')
                time.sleep(1)
                continue
            with state.lock:
                if state.frame is None: continue
                ret, buf = cv2.imencode('.jpg', state.frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.04)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with state.lock: return jsonify(state.data)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files: return jsonify({"error": "No file uploaded"})
    f = request.files['file']
    if f.filename == '': return jsonify({"error": "No file selected"})
    
    path = os.path.join(app.config['VIDEO_FOLDER'], f.filename)
    f.save(path)
    
    report = engine.analyze_video_file(path)
    
    
    
    return jsonify(report)

if __name__ == '__main__':
    threading.Thread(target=video_thread, daemon=True).start()
    threading.Thread(target=audio_listen_thread, daemon=True).start()
    threading.Thread(target=audio_process_thread, daemon=True).start()
    
    print("--- 🎓 AI INTERVIEWER SERVER RUNNING (With DB) ---")
    print("Go to: http://127.0.0.1:5000")
    
    app.run(debug=False, host='0.0.0.0')