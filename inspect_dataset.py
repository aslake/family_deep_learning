__doc__ = """
Script for inspection of image training dataset.

Ver 1.1 -- inspect_dataset.py

Author: Aslak Einbu February 2020.
"""

import numpy as np
import cv2
import os
import config


for person in os.listdir("training_dataset"):
    directory = f'/home/aslei/Documents/fjeslearn/training_dataset/{person}'

    tile_width = config.model_img_dims[1]
    tile_height = config.model_img_dims[0]
    filer = os.listdir(directory)
    antall = len(filer)
    row_tiles = col_tiles = int(np.sqrt(antall))

    tiles = np.zeros((row_tiles * tile_width, col_tiles * tile_height, 3),np.uint8)
    tiles_grey = np.zeros((row_tiles * tile_width, col_tiles * tile_height),np.uint8)
    tiles_grey_norm = np.zeros((row_tiles * tile_width, col_tiles * tile_height),np.uint8)

    row = 0
    col = 0

    for bilde in filer:
        tile = cv2.imread(os.path.join(directory, bilde))
        tile = cv2.resize(tile, (tile_width, tile_height))
        tile_grey = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        tile_grey_norm = (tile_grey - tile_grey.mean())+ 125

        tiles[tile_height*row:tile_height+tile_height*row, tile_width * col:tile_width+tile_width*col] = tile
        tiles_grey[tile_height * row:tile_height + tile_height * row, tile_width * col:tile_width + tile_width * col] = tile_grey
        tiles_grey_norm[tile_height * row:tile_height + tile_height * row, tile_width * col:tile_width + tile_width * col] = tile_grey_norm

        col = col + 1
        if col == col_tiles:
            col = 0
            row = row + 1
        if row == row_tiles:
            row = 0

    cv2.imshow("tiles", tiles)
    cv2.imshow("tiles grayscale", tiles_grey)
    cv2.imshow("tiles grayscale normalized", tiles_grey_norm)

    cv2.waitKey()

