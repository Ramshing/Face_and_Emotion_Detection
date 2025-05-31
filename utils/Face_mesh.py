import cv2
import mediapipe as mp
import time

#cap=cv2.VideoCapture(r"C:\Users\dinyz\Dinesh Ram\Pycharm - Projects\Face Detection and eye blink count\Sample_video\Human face 2.webm")
cap=cv2.VideoCapture(0)
p_time=0

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=5)
drawspec=mpDraw.DrawingSpec(thickness=1,circle_radius=1)


while True:
    SUCCESS, img =cap.read()
    c_time=time.time()
    fps=1/(c_time-p_time)
    p_time=c_time

    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceMesh.process(imgRGB)


    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,facelms,mpFaceMesh.FACEMESH_CONTOURS,drawspec,drawspec)

            for id,lm in enumerate(facelms.landmark):
                #print(id,lm)
                ih,iw,ic=img.shape
                x,y=int(lm.x*iw), int(lm.y*ih)
                print(id,x,y)

    cv2.putText(img,f'FPS:{int(fps)}',(30,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)



