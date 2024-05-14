# Hand Recognition

This project demonstrates real-time hand recognition using the OpenCV library and the HandTrackingModule from the cvzone library. It captures video from a webcam, detects hands, counts the number of fingers extended, and displays the results on the screen.

## Introduction

Hand recognition is a crucial component in various applications such as gesture-based interfaces, sign language interpretation, and virtual reality interactions. This project provides a simple implementation of hand recognition using computer vision techniques.

## Dependencies

To run this project, you need to install the following dependencies:

- Python (version 3.6 or later)
- OpenCV (version 4.x or later)
- cvzone (version 1.5.2 or later)
- NumPy (version 1.19.5 or later)
- tkinter (for GUI, usually included with Python installation)

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

1.Real-time hand detection and finger counting
2.Display of individual finger counts for each hand
3.Calculation of the sum of all finger counts
4.Display of frames per second (FPS) on the video feed
5.Saving captured images with hand overlays

##### License

This project is licensed under the MIT License - see the LICENSE file for details.