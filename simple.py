import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(1)  # Change to 0 if you're using the built-in webcam
detector = HandDetector(maxHands=5, detectionCon=0.2)

offset = 25
imgSize = 480
counter = 0

folder = "Data/5"
if not os.path.exists(folder):
    os.makedirs(folder)

prevTime = 0  # Initialize prevTime for FPS calculation

while True:
    ret, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        fingers = detector.fingersUp(hand)
        finger_count = fingers.count(1)

        # Add finger count text with black background and white text color
        cv2.rectangle(img, (10, 10), (150, 40), (0, 0, 0), -1)
        cv2.putText(img, f"Fingers: {finger_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Calculate FPS
        currentTime = time.time()
        fps = 1 / (currentTime - prevTime)
        prevTime = currentTime  # Update prevTime for the next iteration

        # Display FPS on top-left corner
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('frame', img)
    key = cv2.waitKey(1)
    
    if key == ord("s"):
        counter += 1
        timestamp = time.time()
        file_path = os.path.join(folder, f'Image_{timestamp}.jpg')
        if cv2.imwrite(file_path, imgWhite):
            print(f"Image saved successfully: {folder}")
        else:
            print(f"Error saving image: {folder}")

        print(counter)
    
    if key == 27:  # Press Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()
