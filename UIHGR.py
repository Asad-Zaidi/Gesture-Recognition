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
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Recognition")
        self.root.geometry("300x250")

        self.start_button = tk.Button(
            root, text="Start Recognition", command=self.start_recognition
        )
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(
            root,
            text="Stop Recognition",
            command=self.stop_recognition,
            state="disabled",
        )
        self.stop_button.pack(pady=5)

        self.hand_info_label = tk.Label(root, text="Hand Info:")
        self.hand_info_label.pack(pady=5)

        self.cap = cv2.VideoCapture(1)  # Change to 0 if you're using an internal webcam
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera.")
            self.root.destroy()
            return

        self.detector = HandDetector(maxHands=2, detectionCon=0.2)
        self.offset = 25
        self.imgSize = 480
        self.folder = "Data/5"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.recognition_thread = None
        self.prevTime = 0
        self.running = False

    def start_recognition(self):
        if self.recognition_thread is None or not self.recognition_thread.is_alive():
            self.cap.release()  # Release camera resource before reopening
            self.cap.open(0)  # Reopen the camera
            self.prevTime = time.time()  # Reset prevTime
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.recognition_thread = threading.Thread(target=self.recognize_hand)
            self.recognition_thread.start()
        else:
            messagebox.showinfo("Info", "Recognition is already running.")

    def stop_recognition(self):
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.running = False

    def recognize_hand(self):
        self.running = True
        while self.running:
            ret, img = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture frame from camera.")
                self.running = False
                break

            hands, img = self.detector.findHands(img)
            sum_fingers = 0
            hand_info_text = "Hand Info: \n"
            for i, hand in enumerate(hands):
                x, y, w, h = hand["bbox"]
                fingers = self.detector.fingersUp(hand)
                finger_count = fingers.count(1)
                sum_fingers += finger_count

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"\nFingers: {finger_count}",
                    (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                hand_info_text += f"Hand {i+1}: {finger_count} fingers \n"
            hand_info_text += f"Sum of all counts: {sum_fingers} \n"

            # Calculate FPS
            currentTime = time.time()
            fps = 1 / (currentTime - self.prevTime)
            self.prevTime = currentTime

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
                file_path = os.path.join(self.folder, f"Image_{timestamp}.jpg")
                if cv2.imwrite(file_path, img):
                    messagebox.showinfo(
                        "Image Saved", f"Image saved successfully: {self.folder}"
                    )
                else:
                    messagebox.showerror("Error", f"Error saving image: {self.folder}")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    app = HandRecognitionApp(root)
    root.mainloop()
