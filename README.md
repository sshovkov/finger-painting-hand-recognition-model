# Finger Painting with Hand Tracking

Create digital paintings using your hand movements captured by a webcam. By using the [MediaPipe](https://mediapipe.dev/) for hand tracking.

## Prerequisites

- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- MediaPipe (`pip install mediapipe`)

## How to Run

1. Clone this repository to your local machine.
2. Make sure you have Python and the required libraries installed.
3. Run the following command in your terminal or command prompt to start the finger painting application:

   ```bash
   python finger_painting.py
   ```

## Controls

- **Right hand**: Use your right index finger to draw on the screen.
- **Left hand**: Raise and hold your left hand to stop drawing.
- **Space bar**: Toggle painting mode
- **'q' key**: Quit

## Demo

![finger_painting_small_file](https://github.com/sshovkov/finger-painting-hand-recognition-model/assets/43308603/be6a14b5-6235-4279-beba-ed879c70ec7a)

Hand tracking is implemented using the [MediaPipe](https://mediapipe.dev/) library

![](assets/hand_tracking.gif)
