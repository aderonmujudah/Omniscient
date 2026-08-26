import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

PROFILE_PATH = os.path.expanduser("~/.omniscient/profile.json")

# The mean error threshold in degrees, above which a profile is rejected.
# This is a tunable parameter that depends on the actual measured accuracy on real 
# hardware (S2b). A threshold in degrees cannot be correctly set while the 
# focal length and accuracy measurement itself remain assumed.
# 5.0 degrees is used as a placeholder.
VALIDATION_ERROR_THRESHOLD_DEG = 5.0

def save_profile(profile: Dict[str, Any], path: str = PROFILE_PATH, mean_error_deg: float = 0.0, has_measured_distance: bool = True) -> bool:
    """
    Saves the profile if the validation error is within acceptable limits.
    Returns True if saved, False if rejected.
    """
    if not has_measured_distance:
        logger.warning("Profile rejected: validation rests on an assumed viewing distance.")
        return False
        
    if mean_error_deg > VALIDATION_ERROR_THRESHOLD_DEG:
        logger.warning(f"Profile rejected: mean error {mean_error_deg:.2f} exceeds threshold {VALIDATION_ERROR_THRESHOLD_DEG:.2f}")
        return False
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(profile, f, indent=2)
    return True

def load_profile(path: str = PROFILE_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)
