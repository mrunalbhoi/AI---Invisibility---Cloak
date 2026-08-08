import cv2
import numpy as np
from utils import sort_points

class Effects:

    def __init__(self):
        self.background = None

    def capture_background(self, frame):
        self.background = frame.copy()

    def invisibility(self, frame, points):

        if self.background is None:
            return frame

        if len(points) != 4:
            return frame

        pts = sort_points(points)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        cv2.fillPoly(mask, [pts], 255)

        result = frame.copy()

        result[mask == 255] = self.background[mask == 255]

        cv2.polylines(result, [pts], True, (255, 0, 255), 2)

        return result