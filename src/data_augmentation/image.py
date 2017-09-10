from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import cv2
import numpy as np
import base64
from src.data_augmentation.helpers import convert_yolo_opencv


class Image(object):
    """
    Class to handle images
    """
    def __init__(self, image_path=None, labels=None):
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        self.shape = self.image.shape
        if labels is not {}:
            self.labels = labels

    def random_vertical_flip(self):
        if random.choice([True, False]):  # random flip
            for key, label in self.labels.items():
                self.labels[key]['x'] = 1 - float(self.labels[key]['x'])  # change bounding box coordinates. 1. - x
            self.image = cv2.flip(self.image, 1)

    def random_gaussian_blur(self):
        rnd = random.randrange(1, 8, 2)  # random odd number for gaussian blur
        self.image = cv2.GaussianBlur(self.image, (rnd, rnd), 0)

    def random_brightness(self):
        return self

    def random__pixel_intensity(self):
        return self

    def random_gamma(self):
        gamma = random.uniform(0.5, 1.5)  # random gamma. gamma = 1 renders the original image
        invgamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invgamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self.image = cv2.LUT(self.image, table)

    def display_image(self):
        cv2.imshow("output", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_image_with_labels(self):
        """
        :param image: image loaded with opencv
        :param labels: list of bounding boxes
        :return:
        """
        for label in self.labels.values():
            xmin, ymin, xmax, ymax = convert_yolo_opencv(self.shape, float(label['x']), float(label['y']), float(label['w']), float(label['h']))
            cv2.rectangle(self.image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.imshow("output", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def convert_to_base64(self):
        retval, buffer = cv2.imencode('.jpg', self.image)
        return base64.b64encode(buffer).decode('ascii')
