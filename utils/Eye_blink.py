import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

class BlinkDetection:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        self.detector = FaceMeshDetector(maxFaces=1)
        self.plotY = LivePlot(640, 360, [20, 50], invert=True)
        self.idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
        self.ratioList = []
        self.blinkCounter = 0
        self.counter = 0
        self.color = (255, 0, 255)

    def process_frame(self):
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = self.cap.read()
        if not success:
            return None

        img, faces = self.detector.findFaceMesh(img, draw=False)
        if faces:
            face = faces[0]
            for id in self.idList:
                cv2.circle(img, face[id], 5, self.color, cv2.FILLED)

            ratioAvg = self.calculate_ratio(face)
            self.update_blink_count(ratioAvg)
            cvzone.putTextRect(img, f'Blink Count: {self.blinkCounter}', (10, 50), colorR=self.color)
            imgPlot = self.plotY.update(ratioAvg, self.color)
            img = cv2.resize(img, (640, 360))
            return cvzone.stackImages([img, imgPlot], 2, 1)
        else:
            img = cv2.resize(img, (640, 360))
            return cvzone.stackImages([img, img], 2, 1)

    def calculate_ratio(self, face):
        leftUp, leftDown, leftLeft, leftRight = face[159], face[23], face[130], face[243]
        lengthVer, _ = self.detector.findDistance(leftUp, leftDown)
        lengthHor, _ = self.detector.findDistance(leftLeft, leftRight)
        ratio = int((lengthVer / lengthHor) * 100)
        self.ratioList.append(ratio)
        if len(self.ratioList) > 10:
            self.ratioList.pop(0)
        avg=sum(self.ratioList) / len(self.ratioList)
        print(avg)
        return avg

    def update_blink_count(self, ratioAvg):
        if ratioAvg < 33 and self.counter == 0:
            self.blinkCounter += 1
            self.color = (0, 200, 0)
            self.counter = 1
        if self.counter != 0:
            self.counter += 1
            if self.counter > 10:
                self.counter = 0
                self.color = (255, 0, 255)
        return self.blinkCounter

    def run(self):
        while True:
            imgStack = self.process_frame()
            if imgStack is None:
                break
            cv2.imshow("Blink Detection", imgStack)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    # Replace with path to your video file or use 0 for webcam
    video_path = r"/Sample_video/Human face 2.webm"  # Example: "video.mp4"
    blink_detector = BlinkDetection(video_path)
    blink_detector.run()

if __name__ == "__main__":
    main()