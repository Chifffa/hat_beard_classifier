from typing import Tuple, List, Dict, Union

import numpy as np

from .utils import OnnxModelLoader, preprocess_image


class HatBeardClassifier:
    def __init__(self, model_path: str, input_shape: Tuple[int, int, int]) -> None:
        """
        Class for easy using of hat/beard classifier.

        :param model_path: path to trained model, converted to ONNX format.
        :param input_shape: input shape tuple (height, width, channels).
        """
        self.input_shape = input_shape

        self.model = OnnxModelLoader(model_path)
        self.class_names = ('No hat, no beard', 'Hat', 'Beard', 'Hat and beard')

    def inference(self, image: np.ndarray) -> Dict[str, Union[str, float]]:
        """
        Process image and return class name with probabilities for presence of hat and beard on the image.
        Example of returning dict:
        {
            'class': 'No hat, no beard',
            'hat': 0.05,
            'beard': 0.01
        }

        :param image: image in BGR format (obtained using cv2) to process.
        :return: dict with results.
        """
        img = preprocess_image(image, self.input_shape)
        predictions = self.model.inference(img)
        class_name, hat_prob, beard_prob = self._get_class(predictions)
        return {'class': class_name, 'hat': hat_prob, 'beard': beard_prob}

    def _get_class(self, predictions: List[np.ndarray]) -> Tuple[str, float, float]:
        """
        Get predicted class name and probabilities for each class.

        :param predictions: list of two predicted arrays (hat one-hot and beard one-hot).
        :return: class name and probabilities for each class.
        """
        hat_labels = predictions[0][0]
        beard_labels = predictions[1][0]

        hat_label = int(np.argmax(hat_labels))
        beard_label = int(np.argmax(beard_labels))
        if hat_label == 1 and beard_label == 1:
            return self.class_names[0], hat_labels[0], beard_labels[0]
        elif hat_label == 0 and beard_label == 1:
            return self.class_names[1], hat_labels[0], beard_labels[0]
        elif hat_label == 1 and beard_label == 0:
            return self.class_names[2], hat_labels[0], beard_labels[0]
        else:
            return self.class_names[3], hat_labels[0], beard_labels[0]
