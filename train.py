__doc__ = """
Training of deep convolutional network model.

Ver 1.0 -- train.py
Aslak Einbu Februar 2020.
"""

from keras.callbacks import CSVLogger, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from network import SmallerVGGNet
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os
from train_history import plot_train_history
import time

modellnavn = "einbufjes"

EPOCHS = 400
INIT_LR = 1e-3
BS = 10
IMAGE_DIMS = (96, 96, 1) # 96x96 pixels in greytone

data = []
labels = []

print("Laster inn og preprosseserer bilder...")
imagePaths = sorted(list(paths.list_images("training_dataset")))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image -image.mean() +125
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("Data matrise: {:.2f}MB".format(
    data.nbytes / (1024 * 1000.0)))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

print("Kompilerer modell...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                            depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
csv_logger = CSVLogger(f'model/this/{modellnavn}_history.csv', append=True)
tensorboard_callback = TensorBoard(log_dir=f'model/this/{modellnavn}_board', histogram_freq=1)

print(f'Trener nettverk ({EPOCHS} epochs)...')
start = time.time()

H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1,
    callbacks=[csv_logger, tensorboard_callback])

print(f'Trening av modellen over {EPOCHS} epochs tok {round(float(((time.time() - start)/60)),1)} minutter.')

# save the model to disk
print("Serialiserer nettverk...")
print(f'Lager modell {modellnavn} i katalog modell/this/{modellnavn}.')
model.save(f'model/this/{modellnavn}')

# save the label binarizer to disk
print("Serialserer label binarizer...")
f = open(f'model/this/{modellnavn}_labelbin', "wb")
f.write(pickle.dumps(lb))
f.close()

# plot the training loss and accuracy
print(f'Plotter treningshistorikk for modell {modellnavn}')
plt = plot_train_history(modell=modellnavn)




