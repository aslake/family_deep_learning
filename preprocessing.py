__doc__ = """
Preprocessing of image training dataset and images for prediction.

Ver 1.0 -- preprocessing.py
Aslak Einbu Februar 2020.
"""

import cv2
import config

def preprocess(image):
    """
    Preprocessing of images applied in training and predictions.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image - image.mean() + 125
    image = cv2.resize(image, (config.model_img_dims[1], config.model_img_dims[0]))
    return image