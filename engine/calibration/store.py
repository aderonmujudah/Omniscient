import json
import os
from typing import Dict, Any

PROFILE_PATH = os.path.expanduser("~/.omniscient/profile.json")

def save_profile(profile: Dict[str, Any], path: str = PROFILE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(profile, f, indent=2)

def load_profile(path: str = PROFILE_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)
