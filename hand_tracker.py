import cv2
import mediapipe as mp


class HandTracker:

    def __init__(self):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        self.draw = mp.solutions.drawing_utils

        # Smoothing strength
        self.alpha = 0.5

        # Store previous fingertip positions
        self.previous_points = []

    def smooth_point(self, new_point, old_point):

        if old_point is None:
            return new_point

        x = int(
            old_point[0] * (1 - self.alpha)
            + new_point[0] * self.alpha
        )

        y = int(
            old_point[1] * (1 - self.alpha)
            + new_point[1] * self.alpha
        )

        return (x, y)

    def detect(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb)

        points = []

        if results.multi_hand_landmarks:

            h, w, _ = frame.shape

            for hand in results.multi_hand_landmarks:

                self.draw.draw_landmarks(
                    frame,
                    hand,
                    self.mp_hands.HAND_CONNECTIONS
                )

                # Thumb tip = landmark 4
                thumb = hand.landmark[4]

                # Index fingertip = landmark 8
                index = hand.landmark[8]

                thumb_point = (
                    int(thumb.x * w),
                    int(thumb.y * h)
                )

                index_point = (
                    int(index.x * w),
                    int(index.y * h)
                )

                points.append(thumb_point)
                points.append(index_point)

        # Smooth the points
        smoothed_points = []

        for i, point in enumerate(points):

            if i < len(self.previous_points):

                smooth = self.smooth_point(
                    point,
                    self.previous_points[i]
                )

            else:

                smooth = point

            smoothed_points.append(smooth)

        # Save points for the next frame
        self.previous_points = smoothed_points

        # Draw the smoothed points
        for point in smoothed_points:

             # Outer glow
            cv2.circle(
                frame,
                point,
                12,
                (255, 255, 255),
                2
            )
            # Inner point
            cv2.circle(
                frame,
                point,
                4,
                (255, 255, 255),
                -1
            )

        return frame, smoothed_points