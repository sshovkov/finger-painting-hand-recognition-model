import cv2
import numpy as np
import mediapipe as mp

# Global variables
painting_enabled = True
prev_x, prev_y = 0, 0


def main():
    global prev_x, prev_y, painting_enabled
    # Initialize a hand tracking model using Mediapipe
    # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
    )

    # Drawing variables
    drawing_color = (0, 255, 0)
    landmark_color = (0, 0, 255)
    line_thickness = 10

    # Yse OpenCV for video
    cap = cv2.VideoCapture(0)

    # Get the width and height of the video frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create canvas with the same dimensions as the video frame
    canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    print(
        f"Controls:\n\t- Right Hand: Use your right index finger to draw on the screen.\n\t- Left Hand: Raise and hold your left hand to pause drawing.\n\t- space bar: Toggle painting mode\n\t- q: Quit\n"
    )

    while cap.isOpened():
        frame = cap.read()[1]

        # Flip the frame horizontally for selfie-view
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        results = hands.process(rgb_frame)

        # If a hand is detected
        if results.multi_hand_landmarks and painting_enabled:
            if results.multi_handedness[0].classification[0].label == "Left":
                drawing_color = (0, 255, 0, 0)  # transparent
                reset_prev_coords()
            else:
                drawing_color = (0, 255, 0, 100)  # opaque

                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw red circles on all landmarks
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                    cv2.circle(frame, (x, y), 5, landmark_color, -1)

                # Get index finger tip landmark
                index_finger_tip = hand_landmarks.landmark[8]

                x, y = int(index_finger_tip.x * frame_width), int(
                    index_finger_tip.y * frame_height
                )

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                # Smooth line between previous and current point (interpolation factor 0.1)
                for i in range(1, 10):
                    cur_x = int(prev_x + (x - prev_x) * 0.1 * i)
                    cur_y = int(prev_y + (y - prev_y) * 0.1 * i)

                    cv2.circle(
                        canvas, (cur_x, cur_y), line_thickness, drawing_color, -1
                    )  # -1 is filled circle

                prev_x, prev_y = x, y
        else:
            reset_prev_coords()

        # Combine frame (100% opaque) and canvas (80% opaque)
        result_frame = cv2.addWeighted(frame, 1, canvas, 0.8, 0)

        # Display the resulting frame
        cv2.imshow("Hand Recognition Finger Painting", result_frame)

        # Exit if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # Toggle paint mode with spacebar
        elif key == 32:
            toggle_paint_mode()

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def toggle_paint_mode():
    global painting_enabled, prev_x, prev_y
    painting_enabled = not painting_enabled
    reset_prev_coords()


def reset_prev_coords():
    global prev_x, prev_y
    prev_x, prev_y = 0, 0  # Reset


if __name__ == "__main__":
    main()
