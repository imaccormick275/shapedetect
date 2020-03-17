# import installed libraries
import numpy as np
import cv2 as cv2

from image_pre_processing import preprocess
from shape_detector_opencv import shape_detector

import sys

path = sys.argv[1]

def main():
	image = cv2.imread(path)
	image = preprocess(image, blur=100)
	y_pred = shape_detector(image, min_pixels=85)
	y_pred = tuple(y_pred)
	y_pred = str(y_pred[0]) + ',' + str(y_pred[1]) + ',' + str(y_pred[2])

	return y_pred

if __name__ == "__main__":
    """ This is executed when run from the command line """
    output = main()
    print(output)