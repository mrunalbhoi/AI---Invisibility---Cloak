import cv2
from hand_tracker import HandTracker
from effects import Effects


# Create our tools
tracker = HandTracker()
effects = Effects()


# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# Create fullscreen window
cv2.namedWindow(
    "Invisibility Cloak",
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    "Invisibility Cloak",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


while True:

    # Get frame from webcam
    success, frame = cap.read()

    if not success:
        print("Error: Could not read webcam frame.")
        break

    # Mirror the webcam
    frame = cv2.flip(frame, 1)

    # Detect hands and get fingertip points
    frame, points = tracker.detect(frame)

    # Apply invisibility effect
    frame = effects.invisibility(
        frame,
        points
    )

    # Display instructions
    cv2.putText(
        frame,
        "B = Capture Background | Q = Quit",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Show the frame
    cv2.imshow(
        "Invisibility Cloak",
        frame
    )


    # Keyboard input
    key = cv2.waitKey(1) & 0xFF


    # Capture background
    if key == ord("b"):

        effects.capture_background(frame)

        print("Background Captured!")


    # Quit
    elif key == ord("q"):

        break


# Release webcam
cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()