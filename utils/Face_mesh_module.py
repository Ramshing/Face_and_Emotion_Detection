import cv2
import mediapipe as mp
import time

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


class Face_Mesh_Detector():
    def __init__(self,staticMode=False,maxFaces=2,refine_landmarks=False,minDetectionCon=0.5,minTrackCon=0.5):
        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.refine_landmarks = refine_landmarks
        self.minDetectionCon=minDetectionCon
        self.minTrackCon=minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.refine_landmarks,self.minDetectionCon,self.minTrackCon)
        self.drawspec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.total_blinks = 0
        self.blink_detected = 0
        self.last_blink_time = time.time()
        self.video_capture = None

    def Find_Face_Mesh(self,img,draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, facelms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawspec, self.drawspec)

                face=[]
                for id, lm in enumerate(facelms.landmark):
                    # print(id,lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    print(id, x, y)
                    face.append([x,y])
                faces.append(face)
        return img, len(faces), self.blink_detected


    def reset_blink_count(self):
        self.total_blinks = 0

    def generate_frames(self):
        """Generator function for webcam streaming"""
        if self.video_capture is None:
            self.video_capture = cv2.VideoCapture(0)

        while True:
            success, frame = self.video_capture.read()
            if not success:
                break

            # Process frame for face and eye detection
            processed_frame, _, _ = self.Find_Face_Mesh(frame)

            # Encode and yield the frame
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def process_image(self, image):
        """Process a single image"""
        processed_frame, face_count, blink_count = self.Find_Face_Mesh(image)

        # Save the processed image
        output_path = 'static/uploads/processed_image.jpg'
        cv2.imwrite(output_path, processed_frame)

        return face_count, blink_count

    def process_video(self, video_path):
        """Process a video file"""
        video = cv2.VideoCapture(video_path)

        # Reset blink counter for this video
        self.total_blinks = 0
        total_faces = 0

        # Process each frame
        frame_count = 0

        # Create output video writer
        output_path = 'static/uploads/processed_video.mp4'
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
            processed_frame, faces, _ = self.Find_Face_Mesh(frame)

            # Write frame to output video
            out.write(processed_frame)

            # Update statistics
            total_faces = max(total_faces, faces)
            frame_count += 1

        # Release resources
        video.release()
        out.release()

        return total_faces, self.total_blinks

