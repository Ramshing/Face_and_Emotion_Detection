import eventlet
eventlet.monkey_patch()
from flask import Flask, Response, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
import base64
import time
import mediapipe as mp
from fer import FER
import os
import tempfile

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

class Face_Mesh_Detector:
    def __init__(self, staticMode=False, maxFaces=2, refine_landmarks=False, minDetectionCon=0.5, minTrackCon=0.5):
        self.cap = None
        self.is_streaming = False
        self.blinkCounter = 0
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refine_landmarks = refine_landmarks
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawspec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.last_blink_time = time.time()
        self.eye_status = True
        self.EYE_AR_THRESH = 0.2
        self.ratioList = []
        self.blinkCounter = 0
        self.counter = 0
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.emo_detector = FER(mtcnn=True)

    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.is_streaming = True
            return self.cap.isOpened()
        return True

    def stop_camera(self):
        self.is_streaming = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def Find_Face_Mesh(self, image, draw=True):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result1 = self.emo_detector.detect_emotions(imgRGB)
        faces = []
        emotions = {"primary_emotion": "Unknown", "confidence": 0.0}
        if result1:
            face = result1[0]
            (x, y, w, h) = face['box']
            if draw:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                emotions = face['emotions']
                dominant_emotion = max(emotions, key=emotions.get)
                cv2.putText(image, f"{dominant_emotion}: {emotions[dominant_emotion]:.2f}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                emotions = {"primary_emotion": dominant_emotion, "confidence": emotions[dominant_emotion]}
        results = self.faceMesh.process(imgRGB)
        blink_detected = 0
        if results.multi_face_landmarks:
            for facelms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, facelms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawspec, self.drawspec)
                face = []
                for id, lm in enumerate(facelms.landmark):
                    ih, iw, ic = image.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
                required_indices = [159, 23, 130, 243]
                if all(i < len(face) for i in required_indices):
                    leftUp, leftDown, leftLeft, leftRight = face[159], face[23], face[130], face[243]
                    lengthVer = ((leftUp[0] - leftDown[0]) ** 2 + (leftUp[1] - leftDown[1]) ** 2) ** 0.5
                    lengthHor = ((leftLeft[0] - leftRight[0]) ** 2 + (leftLeft[1] - leftRight[1]) ** 2) ** 0.5
                    if lengthHor != 0:
                        ratio = int((lengthVer / lengthHor) * 100)
                        self.ratioList.append(ratio)
                        if len(self.ratioList) > 10:
                            self.ratioList.pop(0)
                        ratioAvg = sum(self.ratioList) / len(self.ratioList)
                        if ratioAvg < 35 and self.counter == 0:
                            self.blinkCounter += 1
                            blink_detected = 1
                            self.counter = 1
                        if self.counter != 0:
                            self.counter += 1
                            if self.counter > 10:
                                self.counter = 0
        return image, len(faces), emotions, self.blinkCounter

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None, 0, {"primary_emotion": "Unknown", "confidence": 0.0}, 0
        output_path = os.path.join(tempfile.gettempdir(), 'processed_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        out = None
        face_count = 0
        emotions = {"primary_emotion": "Unknown", "confidence": 0.0}
        blink_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame, faces, frame_emotions, frame_blinks = self.Find_Face_Mesh(frame, draw=True)
                face_count = max(face_count, faces)
                if frame_emotions["confidence"] > emotions["confidence"]:
                    emotions = frame_emotions
                blink_count = max(blink_count, frame_blinks)
                if out is None:
                    h, w = frame.shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
                out.write(processed_frame)
            cap.release()
            if out:
                out.release()
            if not os.path.exists(output_path):
                print(f"Error: Output video file {output_path} was not created")
                return None, 0, {"primary_emotion": "Unknown", "confidence": 0.0}, 0
            return output_path, face_count, emotions, blink_count
        except Exception as e:
            print(f"Video processing error: {str(e)}")
            if out:
                out.release()
            return None, 0, {"primary_emotion": "Unknown", "confidence": 0.0}, 0

    def reset_blink_count(self):
        self.blinkCounter = 0

    def generate_frames(self):
        while self.is_streaming and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            processed_frame, face_count, emotions, blink_count = self.Find_Face_Mesh(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            yield frame_b64, face_count, emotions, blink_count
            time.sleep(0.03)

detector = Face_Mesh_Detector()

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    namespace = request.namespace
    print(f'Client {sid} connected to namespace {namespace}')
    emit('stream_status', {'status': 'connected', 'message': 'Connected to backend'}, to=sid, namespace=namespace)

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    namespace = request.namespace
    print(f'Client {sid} disconnected from namespace {namespace}')
    detector.stop_camera()
    emit('stream_status', {'status': 'stopped', 'message': 'Camera stream stopped'}, to=sid, namespace=namespace)

@socketio.on('start_stream')
def handle_start_stream():
    sid = request.sid
    namespace = request.namespace
    if detector.start_camera():
        emit('stream_status', {'status': 'started', 'message': 'Camera stream started'}, to=sid, namespace=namespace)
        def stream_frames(sid, namespace):
            for frame_b64, face_count, emotions, blink_count in detector.generate_frames():
                if not detector.is_streaming:
                    break
                socketio.emit('processed_frame', {
                    'image': f'data:image/jpeg;base64,{frame_b64}',
                    'faces_detected': face_count,
                    'emotion': emotions.get('primary_emotion', 'Unknown'),
                    'confidence': emotions.get('confidence', 0.0),
                    #'blink_count': 0
                }, to=sid, namespace=namespace)
            detector.stop_camera()
            socketio.emit('stream_status', {'status': 'stopped', 'message': 'Camera stream stopped'}, to=sid, namespace=namespace)
        socketio.start_background_task(stream_frames, sid, namespace)
    else:
        emit('stream_status', {'status': 'error', 'message': 'Failed to start camera'}, to=sid, namespace=namespace)

@socketio.on('stop_stream')
def handle_stop_stream():
    sid = request.sid
    namespace = request.namespace
    detector.stop_camera()
    emit('stream_status', {'status': 'stopped', 'message': 'Camera stream stopped'}, to=sid, namespace=namespace)

@app.route('/analyze', methods=['POST'])
def analyze_file():
    sid = request.sid if hasattr(request, 'sid') else None
    namespace = request.namespace if hasattr(request, 'namespace') else '/'
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if not (file.mimetype.startswith('image/') or file.mimetype.startswith('video/')):
        return jsonify({'error': 'Invalid file type'}), 400
    file_data = file.read()
    nparr = np.frombuffer(file_data, np.uint8)
    if file.mimetype.startswith('image/'):
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        processed_image, face_count, emotions, blink_count = detector.Find_Face_Mesh(image, draw=True)
        _, buffer = cv2.imencode('.jpg', processed_image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        response = {
            'image': f'data:image/jpeg;base64,{image_b64}',
            'faces_detected': face_count,
            'emotion': emotions.get('primary_emotion', 'Unknown'),
            'confidence': emotions.get('confidence', 0.0),
            #'blink_count': 0,
            #'file_type': 'image'
        }
        if sid:
            socketio.emit('analysis_result', response, to=sid, namespace=namespace)
        return jsonify(response)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(file_data)
            tmp_file_path = tmp_file.name
        output_path, face_count, emotions, blink_count = detector.process_video(tmp_file_path)
        os.unlink(tmp_file_path)
        if output_path is None:
            return jsonify({'error': 'Failed to process video'}), 500
        try:
            with open(output_path, 'rb') as f:
                video_data = f.read()
            video_b64 = base64.b64encode(video_data).decode('utf-8')
            response = {
                'video': f'data:video/mp4;base64,{video_b64}',
                'faces_detected': face_count,
                'emotion': emotions.get('primary_emotion', 'Unknown'),
                'confidence': emotions.get('confidence', 0.0),
                #'blink_count': 0,
                #'file_type': 'video'
            }
            if sid:
                socketio.emit('analysis_result', response, to=sid, namespace=namespace)
            return jsonify(response)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

@app.route('/reset')
def reset_counter():
    detector.reset_blink_count()
    return jsonify({'success': True, 'blink_count': 0})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)