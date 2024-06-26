# Hand Gestures Recognition

This project demonstrates real-time Hand Gestures Recognition for ASL (American Sign Language) using the OpenCV library and the HandTrackingModule from the cvzone library. It captures video from a webcam, detects hands, counts the number of fingers and Hand Gestures for ASL, and displays the results on the screen.

## Introduction

Hand recognition is a crucial component in various applications such as gesture-based interfaces, sign language interpretation, and virtual reality interactions. This project provides a simple implementation of hand recognition using computer vision techniques.

## Dependencies

To run this project, you need to install the following dependencies:

- Python (version 3.6 or later)
- OpenCV (version 4.6.0.66 or later)
- cvzone (version 1.5.6 or later)
- NumPy (version 1.21.6 or later)
- tkinter (for GUI, usually included with Python installation)
- TensorFlow (version 2.9.1 or later)

You can install these dependencies using pip:

```bash
pip install opencv-python-headless cvzone numpy
```

### USAGE

1.Clone this Repository to your Local Machine.
    git clone <https://github.com/Asad-Zaidi/hand-recognition.git>.
2. Navigate to the project directory:
    cd hand-recognition
3. Run the following command in your terminal:

```bash
python main.py
```

4.Press the "Start Recognition" button to begin hand recognition. The program will use your webcam to capture video and display the hand recognition results in real-time.

5.Press the "Stop Recognition" button to stop the hand recognition process.

#### Features

- Real-time hand detection and finger counting
- Display of individual finger counts for each hand
- Calculation of the sum of all finger counts
- Display of frames per second (FPS) on the video feed
- Saving captured images with hand overlays

## 📝 License <a name="license"></a>

This project is [MIT](./LICENSE) licensed.
