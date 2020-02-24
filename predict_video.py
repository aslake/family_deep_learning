__doc__ = """
Script for face recognition from webcam video stream.

Ver 1.1 -- predict_video.py

Author: Aslak Einbu February 2020.
"""

import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import pickle
import config

modell = config.model

print("Laster modeller...")

# Model for detection of faces
net = cv2.dnn.readNetFromCaffe("model/face_detect/deploy.prototxt.txt",
                               "model/face_detect/res10_300x300_ssd_iter_140000.caffemodel")

# Model for face recognition
model = load_model(f'model/this/{modell}')
lb = pickle.loads(open(f'model/this/{modell}_labelbin', "rb").read())
print(f'Bruker modell: {modell}')

def main():
    """
    Analyses webcam video stream.
    Applies DNN models for detection and recognition of faces.
    """
    camera = cv2.VideoCapture(0)
    print("Analyserer webcam bildestrøm (trykk 'Q' for å stoppe).")

    while True:
        (grabbed, frame) = camera.read()
        if not grabbed:
            break

        frame = imutils.resize(frame, width=1000)
        lager = frame.copy()

        # Detecting faces
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.7:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (255, 255, 255), 2)

            # Recognising faces
            try:
                fjes = lager[startY:endY, startX:endX]
                fjes = cv2.resize(fjes, (100, 120))
                fjes = cv2.cvtColor(fjes, cv2.COLOR_BGR2GRAY)
                fjes = fjes - fjes.mean() + 125
                fjes = cv2.resize(fjes, (config.model_img_dims[1], config.model_img_dims[0]))
                fjes = fjes.astype("float") / 255.0
                fjes = img_to_array(fjes)
                fjes = np.expand_dims(fjes, axis=0)
                proba = model.predict(fjes)[0]
                idx = np.argmax(proba)

                if proba[idx] > 0.8:
                    label = lb.classes_[idx]
                    farge = (0, 0, 0)
                    if proba[idx] < 0.90:
                        label = label + "?"
                    if proba[idx] > 0.97:
                        farge = (0,255,0)

                    #label = "{}: {:.1f} %".format(label, proba[idx] * 100)
                    cv2.putText(frame, label.capitalize(), (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, farge, 3)

                    n = 0
                    for i in range(0, len(lb.classes_)):
                        streng = f'{lb.classes_[i]} : {round(float(100*proba[i]),1)} %'
                        cv2.putText(frame, streng, (startX+10, startY + 40 + n), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        n = n + 20

            except:
                pass

        cv2.imshow("Fjes", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()









