__doc__ = """
Function for plotting of training history of given neural network model.
Saving plot to model folder. Can be applied for monitoring results while model trains.

Ver 1.1 -- train_history.py

Author: Aslak Einbu February 2020.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_train_history(modell="einbufjes"):
    """
    Plot model loss, accuracy and validation loss and accuracy
    as a function of epochs of training
    """
    history = pd.read_csv(f'model/this/{modell}_history.csv')

    plt.style.use("seaborn")
    plt.figure()
    epochs = len(history.epoch)
    plt.plot(np.arange(0, epochs), history["loss"], label="train_loss", color="green")
    plt.plot(np.arange(0, epochs), history["val_loss"], label="val_loss", color="lightgreen")
    plt.plot(np.arange(0, epochs), history["accuracy"], label="train_acc", color="red")
    plt.plot(np.arange(0, epochs), history["val_accuracy"], label="val_acc", color="orange")
    plt.hlines(1.0,0, epochs, colors="black", linestyles="dotted")
    plt.title(f'Trening av modell: {modell}')
    plt.xlabel("Epoch nr")
    plt.ylabel("Loss/Accuracy")
    plt.ylim(0, 3.5)
    plt.xlim(0, epochs)
    plt.legend(loc="upper right")

    xint = []
    locs, labels = plt.xticks()
    for each in locs:
        xint.append(int(each))
    plt.xticks(xint)

    plt.savefig(f'model/this/{modell}.png')
    plt.show()

if __name__ == "__main__":
    plot_train_history()