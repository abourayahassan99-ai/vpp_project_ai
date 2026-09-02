import cv2
import numpy as np

class BlurDetector:
    def __init__(self, threshold: float = 100.0):
        """
        :param threshold: Minimum variance score. Frames below this are rejected.
        """
        self.threshold = threshold

    def calculate_blur_score(self, frame: np.ndarray) -> float:
        """
        Calculates the variance of the Laplacian of a grayscale frame.
        Higher variance indicates sharper focus; lower variance indicates motion blur.
        """
        if frame is None or frame.size == 0:
            return 0.0
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def is_clear(self, frame: np.ndarray) -> bool:
        """
        Returns True if the frame meets or exceeds the clarity threshold.
        """
        return self.calculate_blur_score(frame) >= self.threshold