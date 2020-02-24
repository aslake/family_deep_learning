__doc__ = """
Script for collection of training data for deep learning image recognition.
Saving standardised pictures of detected faces from webcam stream to given folder. 

Ver 1.1 -- collect_faces.py

Author: Aslak Einbu February 2020.
"""

import os
import cv2
import datetime
import imutils
import time
import numpy as np

# Loading neural net model for face detection
net = cv2.dnn.readNetFromCaffe("model/face_detect/deploy.prototxt.txt",
                               "model/face_detect/res10_300x300_ssd_iter_140000.caffemodel")

# Setting folder name for saving of detected images.
person = input("Hvem er personen?")
bildepath = f'/tmp/dnn/{person}'
if not os.path.exists(bildepath):
    os.makedirs(bildepath)
bildepath = f'/tmp/dnn/{person}'


def main():
    """
    Analysing webcam video stream and displaying detected faces.
    Applies deep neural net model for detection of faces in in image.
    Saves images of detected faces to given folder (stops saving after 1000 pictures).
    """
    antall = 0
    sistetid = time.time()
    stdtxt = "Ingen fjes observert!"
    dcttxt = "Fjes observert!"
    noen = False

    camera = cv2.VideoCapture(0)
    print("Analyserer webcam bildestrøm...")
    print(f'Lagrer alle passfoto i {bildepath}.')

    while True:
        (grabbed, frame) = camera.read()
        if not grabbed:
            break

        detekt_txt = stdtxt
        frame = imutils.resize(frame, width=500)
        lager = frame.copy()

        # Detecting faces:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.7:
                continue

            detekt_txt = dcttxt
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.1f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            try:
                fjes = lager[startY:endY, startX:endX]
                fjes = cv2.resize(fjes, (100, 120))

                # Saving image of face
                if (time.time() - sistetid) > 0.5:
                    sistetid = time.time()
                    if antall < 1000:
                        cv2.imwrite(f'{bildepath}/{str(time.time())}.jpg', fjes)
                    antall = antall + 1
                    print(f'\rAntall bilder lagra: {antall}', sep='', end='', flush=True)
            except:
                pass

        noen = True

        if (noen):
            try:
                frame[255:375, 0:100] = fjes
                cv2.putText(frame, "Siste person", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            except:
                pass

        cv2.putText(frame, detekt_txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, datetime.datetime.now().strftime(" %d %B %Y %I:%M:%S%p"), (4, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1)

        cv2.putText(frame, f'Bilder lagra:{antall}', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.imshow("Fjes", frame)
        cv2.moveWindow("Fjes", 1450, 100)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()









