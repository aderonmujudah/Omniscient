import math

# Assumed average adult male/female IPD
PHYSICAL_IPD_MM = 63.0
# Assumed webcam focal length in pixels (for a 640x480 frame, ~65 deg FOV)
CAMERA_FOCAL_LENGTH_PX = 502.0 

def estimate_viewing_distance_mm(ipd_px: float) -> float:
    """Estimates viewing distance in mm from IPD in pixels."""
    if ipd_px <= 0:
        return 600.0 # Default to 60cm if invalid
    return (PHYSICAL_IPD_MM * CAMERA_FOCAL_LENGTH_PX) / ipd_px

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

def validate_calibration(model, test_features: list[tuple[float, float]], test_targets: list[tuple[float, float]], 
                         ipd_px: float, screen_w: int, screen_h: int, diag_mm: float):
    """
    Evaluates the model on previously unseen points.
    Returns (mean_error_deg, worst_error_deg, points_result)
    """
    distance_mm = estimate_viewing_distance_mm(ipd_px)
    
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
    
    return mean_err, worst_err, points
