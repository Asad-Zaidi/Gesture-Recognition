import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20

while True:
    ret, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y-offset:y + h + offset, x - offset:x+w+offset]
        cv2.imshow('ImageCrop', imgCrop)

    cv2.imshow('frame', img)
    cv2.waitKey(1)
