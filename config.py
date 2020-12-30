import os

# Hat/beard classifier parameters.
INPUT_SHAPE = (128, 128, 3)
CLASSIFIER_MODEL_PATH = os.path.join('hat_beard_classifier', 'hat_beard_model.onnx')

# OpenCV face detector parameters.
CASCADE_FILE_PATH = os.path.join('hat_beard_classifier', 'haarcascade_frontalface_alt_tree.xml')
SCALE_FACTOR = 1.1
MIN_NEIGHBOURS = 3

# Increase the size of the found face region by this fraction.
# It is necessary for further accurate hat/beard classification.
COORDINATES_EXTEND_VALUE = 0.2
