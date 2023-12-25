import cv2
import mediapipe as mp
import time
import math
import HandTrackingModule as htm

# sort my closeness to next landmark
# key: 0 value: closest to zero
# POI - points of interest
HAND_SIGNS = {'a': [{'POI': [2, 8, 5, 4, 6, 10, 9, 13, 15, 18, 17, 19]}, {'0': 1, '1': 0, '2': 8, '8': 2, '3': 4, '4': 3, '5': 7, '7': 5, '6': 10, '10': 6, '9': 11, '11': 9, '12': 16, '16': 12, '13': 15, '15': 13, '14': 18, '18': 14, '17': 19, '19': 17, '20': -1, '-1': 20}]}

# what do i quantify as being close
class ASLDecoder:
    def __init__(self, numHands=2):
        '''
        :param numHands: int
        '''
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


        # given a matrix with [landmark id, cx, cy]
        for i in range(len(landmarks)):
            if str(i) not in distanceMap.keys():
                minDis = 10000  # impossible to get
                key = -1
                for j in range(len(landmarks)):
                    if i != j and str(j) not in distanceMap.keys():  # if already located then shortest pair found
                        dis = self.distanceBetween(landmarks[i][1:], landmarks[j][1:])
                        if dis < minDis:
                            minDis = dis
                            key = j
                distanceMap[str(i)] = key # add the location of it's miminum landmark to the dictionary
                distanceMap[str(key)] = i

        # max complexity: O(q*p) where q = 21 and p = 21, 441 passes
        # will optamized to take half as less passes through skipping previously mapped characters
        return distanceMap


    def getSign(self, landmarks):
        closestLandmark = self.findNextLandmark(landmarks)
        for option, key in enumerate(list(HAND_SIGNS.keys())):
            check = len(HAND_SIGNS[key][0]['POI'])
            cashed = 0
            for POI in HAND_SIGNS[key][0]['POI']: # compare points of interest
                if closestLandmark.get(str(POI)) == HAND_SIGNS[key][1].get(str(POI)):
                    cashed += 1
            if cashed == check:
                return key
        return -1
        #return closestLandmark



