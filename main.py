import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

width, height = 1280, 720
fp = "Presentation"

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

pathImages = sorted(os.listdir(fp), key=len)

imageNumber = 0
hs, ws = int(120*2), int(213*2)

detector = HandDetector(detectionCon=0.9, maxHands=1)

buttonPressed = False
buttonCounter = 0
buttonDelay = 35

while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(fp, pathImages[imageNumber])
    imgCurrent = cv2.imread(pathFullImage)

    hands, img = detector.findHands(img, flipType=False)

    cv2.line(img, (0, 450), (width, 450), (0, 255, 0), 10)

    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']

        lmList = hand['lmList']
        indexFinger = lmList[8][0], lmList[8][1]

        xVal = int(np.interp(lmList[8][0], [width//4, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))

        indexFinger = xVal, yVal

        if cy <= 450:

            if fingers == [0, 0, 0, 0, 0]:
                print("Previous Slide")
                buttonPressed = True
                if imageNumber > 0:
                    imageNumber -= 1

            if fingers == [1, 1, 0, 0, 0]:
                print("Next Slide")
                buttonPressed = True
                if imageNumber < len(pathImages)-1:
                    imageNumber += 1

        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    imgSmall = cv2.resize(img,(ws, hs))
    h, w,  _ = imgCurrent.shape
    imgCurrent[0:hs, w-ws:w] = imgSmall

    imS = cv2.resize(imgCurrent, (1280, 720))
    cv2.imshow("PPT Gesture Control - Subhojit Ghosh", imS)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
