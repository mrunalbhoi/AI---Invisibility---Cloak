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


def expand_points(points, expansion=0):

    center_x = sum(p[0] for p in points) / len(points)
    center_y = sum(p[1] for p in points) / len(points)

    expanded = []

    for x, y in points:

        dx = x - center_x
        dy = y - center_y

        distance = math.sqrt(dx * dx + dy * dy)

        if distance == 0:
            expanded.append((int(x), int(y)))
            continue

        new_x = x + (dx / distance) * expansion
        new_y = y + (dy / distance) * expansion

        expanded.append(
            (int(new_x), int(new_y))
        )

    return np.array(expanded, dtype=np.int32)