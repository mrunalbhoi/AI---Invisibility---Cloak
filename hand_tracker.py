import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.draw = mp.solutions.drawing_utils

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                thumb = hand.landmark[4]
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

                cv2.circle(frame, thumb_point, 8, (0,255,0), -1)
                cv2.circle(frame, index_point, 8, (0,0,255), -1)

        return frame, points