# Comprehensive Interview Behavior Analysis Using Multimodal Data

A Flask web application that analyzes interview performance using live camera and microphone input, or an uploaded video. It combines visual and speech-based signals to provide interview feedback.

## Features

- Facial emotion recognition using TensorFlow/Keras
- Eye-contact and posture indicators using MediaPipe
- Speech transcription and sentiment analysis
- Filler-word counting and words-per-minute estimation
- Video-upload analysis
- Local interview-session history using SQLite

## Project Structure

```text
sentiment_webapp/
├── app.py              # Flask application
├── analyzer.py          # Video and audio analysis
├── train_model.py       # Emotion-model training
├── test_accuracy.py     # Model evaluation
├── templates/           # Web pages
└── static/              # Styles and uploaded media
```

## Requirements

- Python 3.10 or later
- Webcam and microphone for live analysis
- A trained emotion model named `ultimate_holistic_model.keras`

Install dependencies:

```bash
pip install flask flask-sqlalchemy opencv-python numpy tensorflow mediapipe SpeechRecognition transformers torch pyaudio
```

## Run the Application

```bash
cd sentiment_webapp
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Training the Emotion Model

The training script expects the FER2013 dataset in this structure:

```text
sentiment_webapp/fer2013/
├── train/
└── test/
```

Run:

```bash
python train_model.py
```

Large datasets, trained models, local session data, and uploads are excluded from the repository.

## Note

This project is for educational and practice purposes. It should not be the only basis for hiring or other high-impact decisions.
