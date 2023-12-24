import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
prevTime = 0
currTime = 0

detector = htm.handDectector()

while True:
    # running webcam
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    # display fps
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, str(int(fps)), (10, 80), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 255), 4)

    # display image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
