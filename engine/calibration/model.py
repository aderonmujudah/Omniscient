import numpy as np

class CalibrationModel:
    def __init__(self):
        self.coeffs_x = None
        self.coeffs_y = None

    def _get_terms(self, fx: float, fy: float) -> list[float]:
        # Second order polynomial: 1, x, y, x*y, x^2, y^2
        return [1.0, fx, fy, fx * fy, fx**2, fy**2]

    def fit(self, features: list[tuple[float, float]], targets: list[tuple[float, float]]) -> None:
        """
        Fits a 2nd order polynomial mapping from feature vector to screen coordinates.
        features: list of (fx, fy)
        targets: list of (tx, ty) screen coordinates
        """
        if len(features) < 5:
            raise ValueError("At least 5 points required for 2nd order polynomial fit.")
            
        A = np.array([self._get_terms(fx, fy) for fx, fy in features])
        tx = np.array([t[0] for t in targets])
        ty = np.array([t[1] for t in targets])
        
        # lstsq returns (solution, residuals, rank, s)
        self.coeffs_x, _, _, _ = np.linalg.lstsq(A, tx, rcond=None)
        self.coeffs_y, _, _, _ = np.linalg.lstsq(A, ty, rcond=None)
        
        # Convert back to standard lists for serialization
        self.coeffs_x = self.coeffs_x.tolist()
        self.coeffs_y = self.coeffs_y.tolist()

    def predict(self, fx: float, fy: float) -> tuple[float, float]:
        """Maps a feature vector to screen coordinates."""
        if self.coeffs_x is None or self.coeffs_y is None:
            raise RuntimeError("Model is not fitted.")
            
        terms = np.array(self._get_terms(fx, fy))
        px = np.dot(terms, self.coeffs_x)
        py = np.dot(terms, self.coeffs_y)
        return float(px), float(py)
        
    def to_dict(self) -> dict:
        return {
            "kind": "poly2",
            "coeffs_x": self.coeffs_x,
            "coeffs_y": self.coeffs_y
        }
        
    def load_dict(self, data: dict) -> None:
        if data.get("kind") != "poly2":
            raise ValueError(f"Unsupported model kind: {data.get('kind')}")
        self.coeffs_x = data["coeffs_x"]
        self.coeffs_y = data["coeffs_y"]
