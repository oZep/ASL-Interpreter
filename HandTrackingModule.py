import cv2
import mediapipe as mp
import time

class handDectector():
    def __init__(self, mode=False, maxHands=2, complexitiy= 1, detectionConf = 0.5, trackConf = 0.5 ):
        '''
        :param mode: bool
        :param maxHands: int
        :param complexitiy: int
        :param detectionConf: int
        :param trackConf: int
        '''
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexitiy
        self.dConf = detectionConf
        self.tConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.dConf, self.tConf)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):
        '''
        :param img:  cap.read()
        :param draw: bool
        :return: img
        '''

        imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # drawing hand points + lines
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS, self.mpDraw.DrawingSpec(color=(255, 0, 0)))

        return img

    def findPosition(self, img, handNum=0, draw=True):
        '''
        :param img:  cap.read()
        :param HandNum: int
        :param draw: bool
        :return: a list with all the landmark positions
        '''
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                # drawing circle over hand landmark
                if draw:
                    cv2.circle(img, (cx,cy), 11, (0,255,255), cv2.FILLED)

        return lmList

def main():
    cap = cv2.VideoCapture(0)
    prevTime = 0
    currTime = 0

    detector = handDectector()

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

if __name__ == "__main__":
    main()