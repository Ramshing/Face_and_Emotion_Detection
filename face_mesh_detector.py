import cv2
import mediapipe as mp
import time
import numpy as np
from utils.Eye_blink import BlinkDetection


class Face_Mesh_Detector():
    def __init__(self, staticMode=False, maxFaces=2, refine_landmarks=False, minDetectionCon=0.5, minTrackCon=0.5):
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

        # Blink detection variables
        #self.total_blinks = 0
        self.last_blink_time = time.time()
        self.eye_status = True  # True for open, False for closed
        self.EYE_AR_THRESH = 0.2
        self.video_capture = None
        self.ratioList = []
        self.blinkCounter = 0
        self.counter = 0

        # Define eye landmarks
        # MediaPipe uses different indices for eyes
        # Left eye landmarks
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Right eye landmarks
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    def calculate_ear(self, landmarks, eye_indices):
        """
        Calculate the eye aspect ratio (EAR) for blink detection
        """
        # Get points
        points = []
        for i in eye_indices:
            points.append([landmarks[i][0], landmarks[i][1]])

        # Calculate horizontal distances
        left_point = points[0]
        right_point = points[8]
        horizontal_dist = np.sqrt((left_point[0] - right_point[0]) ** 2 + (left_point[1] - right_point[1]) ** 2)

        # Calculate vertical distances
        top_left = points[12]
        bottom_left = points[4]
        top_right = points[13]
        bottom_right = points[5]

        v_dist1 = np.sqrt((top_left[0] - bottom_left[0]) ** 2 + (top_left[1] - bottom_left[1]) ** 2)
        v_dist2 = np.sqrt((top_right[0] - bottom_right[0]) ** 2 + (top_right[1] - bottom_right[1]) ** 2)

        # Average vertical distance
        v_dist_avg = (v_dist1 + v_dist2) / 2

        # Calculate EAR
        ear = v_dist_avg / horizontal_dist

        return ear
    def Find_Face_Mesh(self, img, draw=True):

        """
        Process an image to detect face mesh and count blinks
        Returns: processed image, number of faces detected, blink detection in this frame (0 or 1)
        """
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        blink_detected = 0

        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, facelms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawspec,
                                               self.drawspec)

                face = []
                for id, lm in enumerate(facelms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # Remove print to prevent console flooding
                    face.append([x, y])

                faces.append(face)

                # ✅ Check landmark indices before accessing
                required_indices = [159, 23, 130, 243]
                if all(i < len(face) for i in required_indices):
                    leftUp, leftDown, leftLeft, leftRight = face[159], face[23], face[130], face[243]
                    lengthVer = ((leftUp[0] - leftDown[0]) ** 2 + (leftUp[1] - leftDown[1]) ** 2) ** 0.5
                    lengthHor = ((leftLeft[0] - leftRight[0]) ** 2 + (leftLeft[1] - leftRight[1]) ** 2) ** 0.5
                    if lengthHor != 0:  # Prevent division by zero
                        ratio = int((lengthVer / lengthHor) * 100)

                        self.ratioList.append(ratio)
                        if len(self.ratioList) > 10:
                            self.ratioList.pop(0)

                        ratioAvg = sum(self.ratioList) / len(self.ratioList)

                        if ratioAvg < 30 and self.counter == 0:
                            self.blinkCounter += 1
                            blink_detected = 1
                            self.counter = 1

                        if self.counter != 0:
                            self.counter += 1
                            if self.counter > 10:
                                self.counter = 0

                        # Draw info
                        #cv2.putText(img, f'Blink Count: {self.blinkCounter}', (20, 50),
                                    #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(img, f'EAR Avg: {ratioAvg:.2f}', (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


        return img, len(faces), self.blinkCounter

    def reset_blink_count(self):
        """Reset the blink counter"""
        self.blinkCounter = 0

    def generate_frames(self):
        """Generator function for webcam streaming"""
        try:
            # Always create a new capture when starting the stream
            self.video_capture = cv2.VideoCapture(0)

            # Check if camera opened successfully
            if not self.video_capture.isOpened():
                print("Error: Could not open video capture device")
                return

            while True:
                success, frame = self.video_capture.read()
                if not success:
                    print("Failed to grab frame")
                    break

                # Process frame for face and eye detection
                processed_frame, _, blink_count = self.Find_Face_Mesh(frame)

                # Encode and yield the frame
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error in generate_frames: {e}")
            if self.video_capture:
                self.video_capture.release()

    def process_video(self, video_path):
        """Process a video file"""
        try:
            video = cv2.VideoCapture(video_path)

            # Check if video opened successfully
            if not video.isOpened():
                print(f"Error: Could not open video file {video_path}")
                return 0, 0

            # Reset blink counter for this video
            self.reset_blink_count()
            total_faces = 0

            # Process each frame
            frame_count = 0

            # Create output video writer
            output_path = 'static/uploads/processed_video_2.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break

                # Process frame for face and eye detection
                processed_frame, face_count, _ = self.Find_Face_Mesh(frame)

                # Write frame to output video
                out.write(processed_frame)

                # Update statistics
                total_faces = max(total_faces, face_count)
                frame_count += 1

                # Optional: Print progress for long videos
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames")

            # Release resources
            video.release()
            out.release()

            return total_faces, self.blinkCounter

        except Exception as e:
            print(f"Error in process_video: {e}")
            return 0, 0

    def process_image(self, image):
        """Process a single image"""
        try:
            processed_frame, face_count, blink_count = self.Find_Face_Mesh(image)

            # Save the processed image
            output_path = 'static/uploads/processed_image.jpg'
            cv2.imwrite(output_path, processed_frame)

            return face_count, blink_count

        except Exception as e:
            print(f"Error in process_image: {e}")
            return 0, 0