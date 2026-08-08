import cv2
from hand_tracker import HandTracker
from effects import Effects

tracker = HandTracker()
effects = Effects()

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    frame, points = tracker.detect(frame)

    frame = effects.invisibility(frame, points)

    cv2.putText(
        frame,
        "B = Capture Background | Q = Quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Invisibility Cloak", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("b"):
        effects.capture_background(frame)
        print("Background Captured!")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()