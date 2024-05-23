import tkinter as tk
from tkinter import messagebox
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
import threading

class HandRecognitionApp:
    def __init__(abc, root):
        abc.root = root
        abc.root.title("Hand Recognition")
        abc.root.geometry("300x250")

        abc.start_button = tk.Button(
            root, text="Start Recognition", command=abc.start_recognition
        )
        abc.start_button.pack(pady=10)

        abc.stop_button = tk.Button(
            root,
            text="Stop Recognition",
            command=abc.stop_recognition,
            state="disabled",
        )
        abc.stop_button.pack(pady=5)

        abc.hand_info_label = tk.Label(root, text="Hand Info:")
        abc.hand_info_label.pack(pady=5)

        abc.cap = cv2.VideoCapture(0)  # Change to 0 if you're using an internal webcam
        if not abc.cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera.")
            abc.root.destroy()
            return

        abc.detector = HandDetector(maxHands=2, detectionCon=0.2)
        abc.offset = 25
        abc.imgSize = 480
        # abc.folder = "Data/5"
        # if not os.path.exists(abc.folder):
        #     os.makedirs(abc.folder)

        abc.recognition_thread = None
        abc.prevTime = 0
        abc.running = False

    def start_recognition(abc):
        if abc.recognition_thread is None or not abc.recognition_thread.is_alive():
            abc.cap.release()  # Release camera resource before reopening
            abc.cap.open(0)  # Reopen the camera
            abc.prevTime = time.time()  # Reset prevTime
            abc.start_button.config(state="disabled")
            abc.stop_button.config(state="normal")
            abc.recognition_thread = threading.Thread(target=abc.recognize_hand)
            abc.recognition_thread.start()
        else:
            messagebox.showinfo("Info", "Recognition is already running.")

    def stop_recognition(abc):
        abc.start_button.config(state="normal")
        abc.stop_button.config(state="disabled")
        abc.running = False

    def recognize_hand(abc):
        abc.running = True
        while abc.running:
            ret, img = abc.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture frame from camera.")
                abc.running = False
                break

            hands, img = abc.detector.findHands(img)
            sum_fingers = 0
            hand_info_text = "Hand Info: \n"
            for i, hand in enumerate(hands):
                x, y, w, h = hand["bbox"]
                fingers = abc.detector.fingersUp(hand)
                finger_count = fingers.count(1)
                sum_fingers += finger_count

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"Fingers: {finger_count}",
                    (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                # hand_info_text += f"Hand {i+1}: {finger_count} fingers"
                hand_info_text += f" Hand(s) {i+1} "
                hand_info_text += ": "
                hand_info_text += f" Finger(s) {finger_count}"
            hand_info_text += f" Sum of all counts: {sum_fingers}"

            # Calculate FPS
            currentTime = time.time()
            fps = 1 / (currentTime - abc.prevTime)
            abc.prevTime = currentTime

            # Display FPS on frame window
            fps_text = f"FPS: {int(fps)}"

            # Overlay hand info and FPS on the frame
            cv2.rectangle(
                img, (10, 10), (600, 100), (255, 255, 255), -1
            )  # White background
            cv2.putText(
                img,
                hand_info_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

            cv2.rectangle(
                img, (10, 80), (290, 110), (0, 255, 0), -1
            )  # Green background for FPS
            cv2.putText(
                img, fps_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

            cv2.imshow("frame", img)

            key = cv2.waitKey(1)

            if key == ord("s"):
                timestamp = time.time()
                file_path = os.path.join(abc.folder, f"Image_{timestamp}.jpg")
                if cv2.imwrite(file_path, img):
                    messagebox.showinfo(
                        "Image Saved", f"Image saved successfully: {abc.folder}"
                    )
                else:
                    messagebox.showerror("Error", f"Error saving image: {abc.folder}")

        abc.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    app = HandRecognitionApp(root)
    root.mainloop()
