import math
import numpy as np

def sort_points(points):

    center_x = sum(p[0] for p in points) / len(points)
    center_y = sum(p[1] for p in points) / len(points)

    points = sorted(
        points,
        key=lambda p: math.atan2(
            p[1] - center_y,
            p[0] - center_x
        )
    )

    return np.array(points, dtype=np.int32)