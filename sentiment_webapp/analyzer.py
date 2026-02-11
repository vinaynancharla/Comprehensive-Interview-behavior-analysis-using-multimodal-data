import cv2
import numpy as np
import mediapipe as mp
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import speech_recognition as sr
from transformers import pipeline
import re
import json
import zipfile
import tempfile

# --- CONFIGURATION ---
FILLER_WORDS = re.compile(r'\b(um|uh|er|ah|like|okay|right|so|you know)\b', re.IGNORECASE)

class InterviewAnalyzer:
    def __init__(self, model_path, img_size=(48, 48)):
        print("\n--- 🚀 INITIALIZING ULTIMATE INTERVIEW ENGINE v6.0 (Force Mode) ---")
        
        self.img_size = img_size
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # 1. LOAD MODEL
        self.emotion_model = self._load_model_safely(model_path)

        # 2. VISUAL DETECTORS
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )
        self.video_face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, 
            static_image_mode=True,
            min_detection_confidence=0.1
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 3. AUDIO
        print("⏳ Loading NLP...")
        try:
            self.nlp = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            print("✅ NLP Ready")
        except:
            self.nlp = None

        self.recognizer = sr.Recognizer()
        self.mic_active = False
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            self.mic_active = True
            print("✅ Microphone Active")
        except: pass

    def _load_model_safely(self, model_path):
        if not os.path.exists(model_path): return None
        try: return load_model(model_path, compile=False)
        except: pass
        try: return load_model(model_path, compile=False, safe_mode=False)
        except: pass

        print("🔧 Patching Keras Model Config...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(model_path, 'r') as z: z.extractall(temp_dir)
                cfg_path = os.path.join(temp_dir, 'config.json')
                if os.path.exists(cfg_path):
                    with open(cfg_path, 'r') as f: cfg = json.load(f)
                    def clean(c): 
                        if isinstance(c, dict): 
                            c.pop('batch_shape', None)
                            c.pop('time_major', None)
                            for k in c: clean(c[k])
                        elif isinstance(c, list): 
                            for i in c: clean(i)
                    clean(cfg)
                    with open(cfg_path, 'w') as f: json.dump(cfg, f)
                patched = "patched_model.keras"
                with zipfile.ZipFile(patched, 'w', zipfile.ZIP_DEFLATED) as z_out:
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            z_out.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))
                return load_model(patched, compile=False)
        except Exception as e:
            return None

    def analyze_visuals(self, frame, is_video_file=False):
        # 1. Normal Check
        _, stats = self._process_frame_logic(frame, is_video_file)
        
        # 2. Rotation Checks (90, -90, 180)
        if stats["emotion"] == "No Face" and is_video_file:
            for angle in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
                rot_frame = cv2.rotate(frame, angle)
                _, r_stats = self._process_frame_logic(rot_frame, is_video_file)
                if r_stats["emotion"] != "No Face":
                    return rot_frame, r_stats

        # 3. Force Mode (If still No Face)
        if stats["emotion"] == "No Face":
            # Force extract center crop and predict
            h, w, _ = frame.shape
            center_crop = frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)] # Crop middle 60%
            
            if self.emotion_model:
                try:
                    gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, self.img_size)
                    norm = resized / 255.0
                    reshaped = np.reshape(norm, (1, 48, 48, 1))
                    preds = self.emotion_model.predict(reshaped, verbose=0)[0]
                    stats["emotion"] = self.emotions[np.argmax(preds)] + " (Low Conf)"
                    stats["confidence"] = "Low"
                    stats["eye_contact"] = "Unknown"
                    stats["posture"] = "Unknown"
                except:
                    stats["emotion"] = "Neutral (Default)"
            else:
                stats["emotion"] = "Neutral (No Model)"

        return frame, stats

    def _process_frame_logic(self, frame, is_video_file):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance Lighting (Histogram Equalization)
        gray = cv2.equalizeHist(gray)

        stats = {"emotion": "No Face", "posture": "Unknown", "eye_contact": "Unknown", "confidence": "Low"}
        face_found = False
        
        # 1. MediaPipe
        processor = self.video_face_mesh if is_video_file else self.face_mesh
        mesh_results = processor.process(rgb)
        
        if mesh_results.multi_face_landmarks:
            face_found = True
            landmarks = mesh_results.multi_face_landmarks[0]
            
            # Eyes
            l_iris = landmarks.landmark[468]
            l_in = landmarks.landmark[33]
            l_out = landmarks.landmark[133]
            ratio = abs(l_iris.x - l_in.x) / abs(l_out.x - l_in.x)
            stats["eye_contact"] = "Good" if 0.40 < ratio < 0.60 else "Distracted"

            # Emotion
            if self.emotion_model:
                x_lst = [l.x for l in landmarks.landmark]
                y_lst = [l.y for l in landmarks.landmark]
                x1, y1 = int(min(x_lst)*w), int(min(y_lst)*h)
                x2, y2 = int(max(x_lst)*w), int(max(y_lst)*h)
                self._predict_emotion(gray, x1, y1, x2-x1, y2-y1, stats)

        # 2. Haar Fallback
        if not face_found:
            faces = self.face_cascade.detectMultiScale(gray, 1.05, 3) # Lower threshold
            if len(faces) > 0:
                face_found = True
                stats["eye_contact"] = "Good"
                x, y, fw, fh = max(faces, key=lambda i: i[2]*i[3])
                if self.emotion_model: self._predict_emotion(gray, x, y, fw, fh, stats)

        # Posture
        pose_res = self.pose.process(rgb)
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            nose = lm[self.mp_pose.PoseLandmark.NOSE.value].x
            mid = (lm[11].x + lm[12].x) / 2
            stats["posture"] = "Straight" if abs(nose - mid) < 0.08 else "Slouching"

        if face_found:
            pts = 0
            if stats["posture"] == "Straight": pts += 1
            if stats["eye_contact"] == "Good": pts += 1
            if stats["emotion"] in ["Happy", "Neutral", "Angry"]: pts += 1
            stats["confidence"] = ["Low", "Low", "Medium", "High"][pts]
            
        return frame, stats

    def _predict_emotion(self, gray_img, x, y, w, h, stats):
        try:
            pad = 10
            y1, y2 = max(0, y-pad), min(gray_img.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(gray_img.shape[1], x+w+pad)
            roi = gray_img[y1:y2, x1:x2]
            if roi.size == 0: return
            
            resized = cv2.resize(roi, self.img_size)
            norm = resized / 255.0
            reshaped = np.reshape(norm, (1, 48, 48, 1))
            preds = self.emotion_model.predict(reshaped, verbose=0)[0]
            stats["emotion"] = self.emotions[np.argmax(preds)]
        except: pass

    def process_audio(self, audio_data):
        try:
            text = self.recognizer.recognize_google(audio_data)
            fillers = len(FILLER_WORDS.findall(text))
            dur = len(audio_data.frame_data) / (audio_data.sample_rate * audio_data.sample_width)
            wpm = int((len(text.split()) / dur) * 60) if dur > 0 else 0
            
            sent = "Neutral"
            if self.nlp:
                res = self.nlp(text)[0]
                sent = f"{res['label']} ({int(res['score']*100)}%)"
            return text, sent, fillers, wpm
        except: return None, None, 0, 0

    def analyze_video_file(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return {"error": "Invalid Video"}
        
        emo_counts = {}
        conf_counts = {}
        frames = 0
        skip = 5 if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 500 else 1
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames += 1
            if frames % skip != 0: continue
            
            _, stats = self.analyze_visuals(frame, is_video_file=True)
            e, c = stats["emotion"], stats["confidence"]
            
            # Force count whatever we get
            emo_counts[e] = emo_counts.get(e, 0) + 1
            conf_counts[c] = conf_counts.get(c, 0) + 1
        
        cap.release()
        
        # Remove "No Face" from results if we have other data
        if "No Face" in emo_counts and len(emo_counts) > 1:
            del emo_counts["No Face"]
            
        dom_emo = max(emo_counts, key=emo_counts.get) if emo_counts else "Neutral"
        avg_conf = max(conf_counts, key=conf_counts.get) if conf_counts else "Low"
        
        return {
            "dominant_emotion": dom_emo,
            "overall_confidence": avg_conf,
            "frames_analyzed": frames,
            "status": "Complete"
        }