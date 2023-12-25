import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import ASLHandDecoder as asl

cap = cv2.VideoCapture(0)
prevTime = 0
currTime = 0

detector = htm.HandDectector()
reader = asl.ASLDecoder()

word = ''

while True:
    # running webcam
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    letter = reader.getSign(lmList)
    if letter != -1:
        word += letter

    print(word)

    cv2.putText(img, str(word), (30, 80), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 4)

    # display fps
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
