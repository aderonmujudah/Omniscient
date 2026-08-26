import math

# Source: Gordon, C.C. et al. (2012). 2012 Anthropometric Survey of U.S. Army Personnel (ANSUR II). 
# Male mean interpupillary distance is 64.7 mm (SD = 3.7 mm), female mean is 62.3 mm (SD = 3.6 mm). 
# A combined generic mean of 63.5 mm is used.
PHYSICAL_IPD_MM = 63.5

# Assumed typical webcam horizontal field of view in degrees.
ASSUMED_HFOV_DEG = 65.0

def get_focal_length_px(frame_width_px: int) -> float:
    """Derives focal length from an assumed horizontal FOV and actual frame width."""
    hfov_rad = math.radians(ASSUMED_HFOV_DEG)
    return (frame_width_px / 2.0) / math.tan(hfov_rad / 2.0)

def estimate_viewing_distance_mm(ipd_px: float, frame_width_px: int) -> float:
    """Estimates viewing distance in mm from IPD in pixels."""
    if ipd_px <= 0:
        return None
    focal_length_px = get_focal_length_px(frame_width_px)
    return (PHYSICAL_IPD_MM * focal_length_px) / ipd_px

def compute_pixel_error(target: tuple[float, float], predicted: tuple[float, float]) -> float:
    return math.hypot(predicted[0] - target[0], predicted[1] - target[1])

def pixel_to_degrees(error_px: float, distance_mm: float, screen_w: int, screen_h: int, diag_mm: float) -> float:
    """Converts a pixel error to degrees of visual angle."""
    if diag_mm <= 0:
        return 0.0
        
    diag_px = math.hypot(screen_w, screen_h)
    px_per_mm = diag_px / diag_mm
    
    error_mm = error_px / px_per_mm
    
    # tan(theta) = error_mm / distance_mm
    theta_rad = math.atan2(error_mm, distance_mm)
    return math.degrees(theta_rad)

import logging
logger = logging.getLogger(__name__)

def validate_calibration(model, test_features: list[tuple[float, float]], test_targets: list[tuple[float, float]], 
                         ipd_px: float, frame_width_px: int, screen_w: int, screen_h: int, diag_mm: float):
    """
    Evaluates the model on previously unseen points.
    Returns (mean_error_deg, worst_error_deg, points_result, has_measured_distance)
    """
    has_measured_distance = True
    distance_mm = estimate_viewing_distance_mm(ipd_px, frame_width_px)
    
    if distance_mm is None:
        has_measured_distance = False
        logger.warning("Absent IPD measurement. Assumed viewing distance of 600.0 mm used, resulting error in degrees is fabricated.")
        distance_mm = 600.0
    
    errors_deg = []
    points = []
    
    for (fx, fy), (tx, ty) in zip(test_features, test_targets):
        px, py = model.predict(fx, fy)
        err_px = compute_pixel_error((tx, ty), (px, py))
        err_deg = pixel_to_degrees(err_px, distance_mm, screen_w, screen_h, diag_mm)
        errors_deg.append(err_deg)
        points.append({
            "target": [tx, ty],
            "predicted": [px, py],
            "error_deg": err_deg
        })
        
    mean_err = sum(errors_deg) / len(errors_deg) if errors_deg else 0.0
    worst_err = max(errors_deg) if errors_deg else 0.0
    
    return mean_err, worst_err, points, has_measured_distance
