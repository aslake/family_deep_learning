__doc__ = """
Script for face recognition of images in given folder.
TODO: Plot total set as tiled page...

Ver 1.1 -- predict_folder.py

Author: Aslak Einbu February 2020.
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os
import time

modell = "einbufjes"

directory = r"/home/aslei/Documents/fjeslearn/training_dataset/kari"
#directory = r"/home/aslei/Documents/fjeslearn/test_dataset"

print("Loading network...")
model = load_model(f'model/this/{modell}')
lb = pickle.loads(open(f'model/this/{modell}_labelbin', "rb").read())
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        print(os.path.join(directory, filename))

        # load the image
        image = cv2.imread(os.path.join(directory, filename))
        output = image.copy()

        # pre-process the image for classification
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (96, 96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # classify the input image
        print("Classifying image...")
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx]

        label = "{}: {:.2f} %".format(label, proba[idx] * 100)
        output = imutils.resize(output, width=400)
        #cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        #            0.7, (0, 255, 0), 2)
        n = 0
        for i in range(0, len(lb.classes_)):
            streng = f'{lb.classes_[i]} : {round(float(100 * proba[i]), 1)} %'
            cv2.putText(output, streng, (10,40 + n), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            n = n + 20

        print("Bilde av {}".format(label))
        cv2.imshow("Output", output)

        cv2.waitKey(0)

