import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize webcam
cap = cv2.VideoCapture(0)
# Initialize hand detector with max 1 hand detection
detector = HandDetector(maxHands=1)

# Parameters
offset = 25
imgSize = 480
counter = 0

# Create folder to save images if it doesn't exist
folder = "Data/C"
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    ret, img = cap.read()  # Read frame from webcam
    if not ret:
        print("Failed to grab frame")
        break

    hands, img = detector.findHands(img)  # Detect hands
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]  # Get bounding box of hand
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Create white image

        # Ensure the crop area is within the image bounds
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        imgCropShape = imgCrop.shape

        if imgCropShape[0] > 0 and imgCropShape[1] > 0:
            aspectRatio = h / w

            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Resize maintaining aspect ratio
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize  # Center the resized image
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Display images
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)
            except cv2.error as e:
                print(f"Resize error: {e}")

    cv2.imshow("frame", img)  # Display original frame
    key = cv2.waitKey(1)
    
    if key == ord("s"):
        counter += 1
        timestamp = time.time()
        file_path = os.path.join(folder, f"Image_{timestamp}.jpg")
        if cv2.imwrite(file_path, imgWhite):
            print(f"Image saved successfully: {file_path}")
        else:
            print(f"Error saving image: {file_path}")
        print(counter)

cap.release()
cv2.destroyAllWindows()
