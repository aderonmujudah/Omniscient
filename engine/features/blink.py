import math
from engine.sources.base import Point2D

def compute_ear(top: Point2D, bottom: Point2D, inner: Point2D, outer: Point2D, w: int, h: int) -> float:
    """
    Computes the Eye Aspect Ratio (EAR) in pixel space.
    EAR = |top - bottom| / |outer - inner|
    """
    def pixel_distance(p1: Point2D, p2: Point2D) -> float:
        return math.hypot((p2.x - p1.x) * w, (p2.y - p1.y) * h)

    v_dist = pixel_distance(top, bottom)
    h_dist = pixel_distance(inner, outer)
    return v_dist / h_dist if h_dist > 0 else 0.0
