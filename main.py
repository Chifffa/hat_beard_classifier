import os
import argparse
from typing import Union

import cv2

from hat_beard_classifier import HatBeardClassifier, SimpleFaceDetector, draw_results, get_coordinates
from config import (
    INPUT_SHAPE, CLASSIFIER_MODEL_PATH, CASCADE_FILE_PATH, SCALE_FACTOR, MIN_NEIGHBOURS, COORDINATES_EXTEND_VALUE
)


def process_video(video_path: Union[int, str]) -> None:
    """
    Process video from file or webcam and show it frame by frame. Press "q" to exit.

    :param video_path: path to video file or webcam index.
    """
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, image = cap.read()
        if not ret:
            print('Can\'t get frame. Stop working.')
            cap.release()
            return
        faces = detector.inference(image)
        classes = []
        for face_coordinates in faces:
            x, y, w, h = get_coordinates(image, face_coordinates, COORDINATES_EXTEND_VALUE)
            class_result = classifier.inference(image[y:y + h, x:x + w, :])
            classes.append(class_result)
        image = draw_results(image, faces, classes)
        cv2.imshow('Video', image)
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            return


def process_images(images_path: str, use_detector: bool) -> None:
    """
    Process all images from folder and show it frame by frame. Press any key to continue, press "q" to exit.

    :param images_path: path to folder with images.
    :param use_detector: if False, then don't use face detector and classify whole image.
    """
    images_paths = [os.path.join(images_path, p) for p in os.listdir(images_path)]
    for img_path in images_paths:
        image = cv2.imread(img_path)
        if image is None:
            print('Can\'t read image: "{}".'.format(img_path))
            continue
        if use_detector:
            faces = detector.inference(image)
            classes = []
            for face_coordinates in faces:
                x, y, w, h = get_coordinates(image, face_coordinates, COORDINATES_EXTEND_VALUE)
                class_result = classifier.inference(image[y:y + h, x:x + w, :])
                classes.append(class_result)
            image = draw_results(image, faces, classes)
        else:
            class_result = classifier.inference(image)
            image = draw_results(image, [[0, image.shape[0] - 1, 0, 0]], [class_result])
        cv2.imshow('Video', image)
        if cv2.waitKey(0) == ord('q'):
            return


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser(
        'Script for processing images or video to find faces and find headdress and beard on it.'
    )
    parser.add_argument('-v', '--video', type=str, default='0', help='Path to video file or any webcam index.')
    parser.add_argument(
        '--images', type=str, default=None,
        help='Path to folder with images. If provided then this images will be processed and visualized.'
    )
    parser.add_argument('--no_detector', action='store_true', help='Don\'t use face detector while images processing.')
    return parser.parse_args()


if __name__ == '__main__':
    detector = SimpleFaceDetector(CASCADE_FILE_PATH, SCALE_FACTOR, MIN_NEIGHBOURS)
    classifier = HatBeardClassifier(CLASSIFIER_MODEL_PATH, INPUT_SHAPE)

    args = parse_args()
    if args.images is None:
        vid_path = args.video
        if not os.path.exists(vid_path):
            vid_path = int(vid_path)
        process_video(vid_path)
    else:
        process_images(args.images, not args.no_detector)
