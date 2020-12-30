from typing import List, Tuple, Dict, Union

import cv2
import numpy as np
from onnxruntime import InferenceSession


class OnnxModelLoader:
    def __init__(self, onnx_path: str) -> None:
        """
        Class for loading ONNX models to inference on CPU. CPU inference is very effective using onnxruntime.

        :param onnx_path: path to ONNX model file (*.onnx file).
        """
        self.sess = InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        self.input_name = [x.name for x in self.sess.get_inputs()][0]
        self.output_names = [x.name for x in self.sess.get_outputs()]

    def inference(self, inputs: np.ndarray) -> List[np.ndarray]:
        """
        Run inference.

        :param inputs: list of arguments, order must match names in input_names.
        :return: list of outputs.
        """
        return self.sess.run(self.output_names, input_feed={self.input_name: inputs})


def preprocess_image(image: np.ndarray, input_shape: Tuple[int, int, int], bgr_to_rgb: bool = True) -> np.ndarray:
    """
    Copy input image and preprocess it for further inference.

    :param image: image numpy array in RGB or BGR format.
    :param input_shape: input shape tuple (height, width, channels).
    :param bgr_to_rgb: if True, then convert image from BGR to RGB.
    :return: image array ready for inference.
    """
    img = image.copy()
    if bgr_to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape[:2][::-1], interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img / 255.0, axis=0)
    return np.float32(img)


def draw_results(image: np.ndarray, faces: List[List[int]],
                 hats_beards: List[Dict[str, Union[str, float]]]) -> np.ndarray:
    """
    Draw found face and predicted image class on original image.

    :param image: original image.
    :param faces: list with faces coordinates.
    :param hats_beards: list with classification results for each face respectively.
    :return: image with bounding boxes.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (0, 0, 255)
    for (x, y, w, h), hat_beard_dict in zip(faces, hats_beards):
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = '{}. hat = {:.02f}%, beard = {:.02f}%'.format(
            hat_beard_dict['class'], hat_beard_dict['hat'] * 100, hat_beard_dict['beard'] * 100
        )
        (label_width, label_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        if x + label_width > image.shape[1]:
            _image = np.zeros((image.shape[0], x + label_width, image.shape[2]), dtype=np.uint8)
            _image[:, :image.shape[1], :] = image
            image = _image
        cv2.rectangle(image, (x, y - label_height - baseline), (x + label_width, y), color, -1)
        cv2.putText(image, text, (x, y - baseline // 2), font,
                    font_scale, (0, 0, 0), lineType=cv2.LINE_AA, thickness=thickness)
    return image


def get_coordinates(image: np.ndarray, coordinates: List[int], extend_value: float) -> Tuple[int, int, int, int]:
    """
    Get extended coordinates of found face for accurate hat/beard classification.

    :param image: original image.
    :param coordinates: found face coordinates in format [x, y, w, h].
    :param extend_value: positive float < 1.
    :return: obtained coordinates in same format.
    """
    x, y, w, h = coordinates
    x = int(np.clip(x - extend_value * w, 0, image.shape[1]))
    y = int(np.clip(y - extend_value * h, 0, image.shape[0]))
    w = int(np.clip(w * (1 + extend_value), 0, image.shape[1]))
    h = int(np.clip(h * (1 + extend_value), 0, image.shape[0]))
    return x, y, w, h
