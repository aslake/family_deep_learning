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
import config
from preprocessing import preprocess

modell = config.model

directory = r"/home/aslei/Documents/fjeslearn/training_dataset/kari"
#directory = r"/home/aslei/Documents/fjeslearn/test_dataset"

print("Loading network...")
model = load_model(f'model/this/{modell}')
lb = pickle.loads(open(f'model/this/{modell}_labelbin', "rb").read())

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        print(os.path.join(directory, filename))

        # load the image and preprocess for prediction
        image = cv2.imread(os.path.join(directory, filename))
        output = image.copy()

        image = preprocess(image)
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
            cv2.putText(output, streng, (10,40 + n), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            n = n + 20

        print("Bilde av {}".format(label))
        cv2.imshow("Output", output)

        cv2.waitKey(0)

