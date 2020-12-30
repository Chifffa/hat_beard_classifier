from typing import List

import numpy as np
import cv2


class SimpleFaceDetector:
    def __init__(self, cascade_file_path: str, scale_factor: float, min_neighbors: int) -> None:
        """
        Wrapper for face detection using Haar cascade.

        :param cascade_file_path: path to *.xml file with Haar cascade data.
        :param scale_factor: scale_factor for CascadeClassifier, float > 1.
        :param min_neighbors: min_neighbors for CascadeClassifier, int > 0.
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

        self.detector = cv2.CascadeClassifier(cascade_file_path)

    def inference(self, image: np.ndarray) -> List[List[int]]:
        """
        Detect faces in image using Haar cascade.

        :param image: image in BGR format (obtained using cv2).
        :return: list of found faces (each one is [x, y, w, h] - bounding box coordinates).
        """
        img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        predictions = self.detector.detectMultiScale(img, self.scale_factor, self.min_neighbors,
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
        return [list(p) for p in predictions]
