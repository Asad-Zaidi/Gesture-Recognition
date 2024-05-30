import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize the video capture, hand detector, and classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Constants
offset = 20
imgSize = 300

# labels = ["A", "B", "C"]

# Read labels from text file
with open("Model/labels.txt", "r") as file:
    labels = [line.strip() for line in file]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure cropping is within the image bounds
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]
        imgCropShape = imgCrop.shape

        if imgCropShape[0] == 0 or imgCropShape[1] == 0:
            print("Invalid crop dimensions")
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            if imgResize.shape[1] > imgSize:  # Ensure resized width fits
                imgResize = cv2.resize(imgResize, (imgSize, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            if imgResize.shape[0] > imgSize:  # Ensure resized height fits
                imgResize = cv2.resize(imgResize, (imgSize, imgSize))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        label = labels[index]  # Get the label from the labels list
        print(prediction, index, label)  # Debug print statement

        cv2.rectangle(
            imgOutput,
            (x - offset, y - offset - 50),
            (x - offset + 90, y - offset - 50 + 50),
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.putText(
            imgOutput,
            label,
            (x, y - 26),
            cv2.FONT_HERSHEY_COMPLEX,
            1.7,
            (255, 255, 255),
            2,
        )
        cv2.rectangle(
            imgOutput,
            (x - offset, y - offset),
            (x + w + offset, y + h + offset),
            (0, 255, 0),
            4,
        )
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 
