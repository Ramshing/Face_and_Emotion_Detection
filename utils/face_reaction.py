import cv2
from fer import FER
import matplotlib.pyplot as plt
import moviepy.editor


def detect_emotions(image_path):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize the FER detector
    detector = FER(mtcnn=True)

    # Detect emotions
    result = detector.detect_emotions(img)

    # Process and display results
    for face in result:
        # Get bounding box
        (x, y, w, h) = face['box']

        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get emotions
        emotions = face['emotions']
        dominant_emotion = max(emotions, key=emotions.get)

        # Display dominant emotion
        cv2.putText(img, f"{dominant_emotion}: {emotions[dominant_emotion]:.2f}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Print all emotions
        print("Detected emotions:")
        for emotion, score in emotions.items():
            print(f"{emotion}: {score:.2f}")

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('emotion_detection_result.png')
    plt.close()

def live_emotion_detection():
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_emotions(rgb_frame)

        for face in results:
            (x, y, w, h) = face['box']
            emotions = face['emotions']
            dominant_emotion = max(emotions, key=emotions.get)

            # Draw bounding box and emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{dominant_emotion}: {emotions[dominant_emotion]:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Print all emotions
            print("Detected emotions:")
            for emotion, score in emotions.items():
                print(f"{emotion}: {score:.2f}")

        cv2.imshow("Live Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

def video_emotion_detection(video_path):
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_emotions(rgb_frame)

        for face in results:
            (x, y, w, h) = face['box']
            emotions = face['emotions']
            dominant_emotion = max(emotions, key=emotions.get)

            # Draw bounding box and emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{dominant_emotion}: {emotions[dominant_emotion]:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Print all emotions
            print("Detected emotions:")
            for emotion, score in emotions.items():
                print(f"{emotion}: {score:.2f}")

        cv2.imshow("Video Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Replace with your image path
    #image_path = r"C:\Users\dinyz\Dinesh Ram\Pycharm - Projects\Face Detection and eye blink count\static\uploads\sample_img.jpg"
    #detect_emotions(image_path)
    live_emotion_detection()
    #video_emotion_detection(r"C:\Users\dinyz\Dinesh Ram\Pycharm - Projects\Face Detection and eye blink count\static\uploads\human face sample.webm")