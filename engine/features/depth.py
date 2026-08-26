import math
from engine.sources.base import Point2D

def compute_ipd_px(left_iris: Point2D, right_iris: Point2D, w: int, h: int) -> float:
    """
    Computes inter-pupillary distance in image pixels.
    """
    return math.hypot((right_iris.x - left_iris.x) * w, (right_iris.y - left_iris.y) * h)
