## Virtual Keyboard using OpenCV, MediaPipe, and PyInput
This project implements a virtual keyboard using computer vision techniques. It leverages the OpenCV library for image processing, MediaPipe for hand landmark detection, and PyInput for simulating keyboard presses. The virtual keyboard is displayed on the screen, and key presses are detected based on hand landmarks.

## Requirements
Python 3.7+
OpenCV
MediaPipe
PyInput
NumPy


## Usage
The virtual keyboard detects the position of your index and middle fingers to determine which key to press. When your index finger (landmark 8) hovers over a key and the distance between your index and middle fingers (landmark 12) is less than 30 pixels, the key is pressed.
 
