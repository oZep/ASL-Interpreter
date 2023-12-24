import cv2
import mediapipe as mp
import time
import math

# sort my closeness to next landmark
# key: 0 value: closest to zero

HAND_SIGNS = {'a': {'POI': [0,1,2,4,6,7,8,10], '0': 12, '1': 2, '2': 4, '3': 2,
                    '4': 6, '5': -1, '6': 10,'7': 11, '8': 12, '9': -1,
                    '10': 6, '11': 15, '12': 0, '13': -1, '14': 18, '15': -1,
                    '16': 17, '18': 15, '19': 15, '20': 14}}


# what do i quantify as being close
class ASLDecoder:
    def __init__(self, numHands=2):
        self.numHands = numHands

    def distanceBetween(self, p1, p2):
        '''
        :param p1: List of pos
        :param p2: List of pos
        :return: distance
        '''
        dx = math.pow(p1[0] - p2[0], 2)
        dy = math.pow(p1[1] -p2[1], 2)
        dis = math.sqrt(dx + dy)
        return dis

    def findNextLandmark(self, landmarks):
        '''
        :param landmarks: List of all landmarks
        :return: an ordered dic with the value: landmark, key: closest neighbor
        '''
        distanceMap = {} # to optamize

        p1 = []
        p2 = []
        minDis = 10000 # impossible to get
        key = -1
        # given a matrix with [landmark id, cx, cy]
        for i in len(landmarks):
            if str(i) not in distanceMap.keys():
                p1.append(landmark[i][1])
                p1.append(landmark[i][2])
                for j in len(landmarks):
                    if str(j) not in distanceMap.keys():  # if already located then shortest pair found
                        p2.append(landmarks[j][1])
                        p2.append(landmarks[j][2])
                        dis = self.distanceBetween(p1, p2)
                        if dis < minDis:
                            minDis = dis
                            key = j
                distanceMap[str(i)] = j # add the location of it's miminum landmark to the dictionary
                distanceMap[str(j)] = i

        # max complexity: O(q*p) where q = 21 and p = 21, 441 passes
        # will optamized to take half as less passes through skipping previously mapped characters
        return distanceMap


    def getSign(self, landmarks):
        self.findNextLandmark(landmarks)
