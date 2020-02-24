__doc__ = """
Function for plotting of training history of given neural network model.
Saving plot to model folder. Can be applied for monitoring results while model trains.

Ver 1.1 -- train_history.py

Author: Aslak Einbu February 2020.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config


def plot_train_history(modell=config.model):
    """
    Plot model loss, accuracy and validation loss and accuracy
    as a function of epochs of training
    """
    history = pd.read_csv(f'model/this/{modell}_history.csv')
    epochs = len(history.epoch)

    plt.style.use("ggplot")
    plt.rcParams['figure.figsize'] = (5, 9)
    plt.plot(np.arange(0, epochs), history["accuracy"], label="model accuracy", color="red", zorder=10, linewidth=2)
    plt.plot(np.arange(0, epochs), history["loss"], label="training loss", color="blue", zorder=9, linewidth=2)
    plt.plot(np.arange(0, epochs), history["val_accuracy"], label="validation accuracy", color="red", zorder=1, linewidth=1, alpha= 0.4)
    plt.plot(np.arange(0, epochs), history["val_loss"], label="validation loss", color="blue", zorder=2, linewidth=1, alpha= 0.4)
    plt.hlines(1.0,0, epochs, colors="black", linestyles="dotted")
    plt.title(f'Trening av modell: {modell}')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy / Loss")
    plt.ylim(0, 2.)
    plt.yticks(np.append(np.arange(0, 1., 0.05), (np.arange(1, 2., 0.2) )))

    plt.xlim(0, epochs)
    plt.legend(loc="upper right")
    plt.tight_layout(True)

    xint = []
    locs, labels = plt.xticks()
    for each in locs:
        xint.append(int(each))
    plt.xticks(xint)

    plt.savefig(f'model/this/{modell}.png')
    plt.show()

if __name__ == "__main__":
    plot_train_history()