from engine.sources.base import EyeGeometry

def extract_features(left_eye: EyeGeometry, right_eye: EyeGeometry) -> tuple[float, float]:
    """
    Extracts the normalized feature vector from eye geometry.
    Returns (fx, fy), the average normalized iris displacement across both eyes.
    fx = (iris.x - (inner.x + outer.x)/2) / |outer.x - inner.x| (Wait, the spec says |outer - inner| distance, not axis distance)
    """
    import math

    def get_displacement(eye: EyeGeometry) -> tuple[float, float]:
        # Inter-corner distance as the scale reference for BOTH axes.
        scale = math.hypot(eye.outer.x - eye.inner.x, eye.outer.y - eye.inner.y)
        if scale == 0:
            return 0.0, 0.0
            
        cx = (eye.inner.x + eye.outer.x) / 2.0
        cy = (eye.inner.y + eye.outer.y) / 2.0
        
        dx = (eye.iris.x - cx) / scale
        dy = (eye.iris.y - cy) / scale
        
        return dx, dy

    l_dx, l_dy = get_displacement(left_eye)
    r_dx, r_dy = get_displacement(right_eye)
    
    # Average two independent estimates
    return (l_dx + r_dx) / 2.0, (l_dy + r_dy) / 2.0
