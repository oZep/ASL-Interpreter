import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

while True:
    # running webcam
    success, img = cap.read()
    imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)

                # drawing circle over hand landmark
                # cv2.circle(img, (cx,cy), 16, (0,255,255), cv2.FILLED)

            # drawing hand points + lines
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, mpDraw.DrawingSpec(color=(255, 0, 0)))


    # display fps
    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, str(int(fps)), (10,80), cv2.FONT_HERSHEY_DUPLEX, 3, (0,255,255), 4)


    # display image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc to exit the loop
        break


cap.release()
cv2.destroyAllWindows()
