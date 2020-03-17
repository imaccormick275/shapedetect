# import libraries
import cv2
import numpy as np

from helpers import resize
from helpers import grab_contours

class ShapeDetector:
    def __init__(self):
        pass
    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return the name of the shape
        return shape

def shape_detector(image, min_pixels=85, return_img=False):

    # load the image and resize it to a smaller factor so that
    # the shapes can be approximated better
    resized = resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    sd = ShapeDetector()
    
    
       
    # variables to count shapes
    num_cir = 0
    num_tri = 0
    num_sqr = 0
    unkn = 0

    # loop over the contours
    if return_img:
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_pixels:
                M = cv2.moments(c)
                cX = int((M["m10"] / (M["m00"] + 1e-7)) * ratio)
                cY = int((M["m01"] / (M["m00"] + 1e-7)) * ratio)
                shape = sd.detect(c)
                # multiply the contour (x, y)-coordinates by the resize ratio,
                # then draw the contours and the name of the shape on the image
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)


                # count shapes
                if shape =='circle':
                    num_cir += 1
                elif shape =='triangle':
                    num_tri += 1
                elif shape=='square':
                    num_sqr += 1
                else:
                    unkn += 1

            num_shapes = np.array([num_sqr, num_cir, num_tri])

        return image, num_shapes
    else:
        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_pixels:
                shape = sd.detect(c)
                area = cv2.contourArea(c)

                # count shapes
                if shape =='circle':
                    num_cir += 1
                elif shape =='triangle':
                    num_tri += 1
                elif shape=='square':
                    num_sqr += 1
                else:
                    unkn += 1

        num_shapes = np.array([num_sqr, num_cir, num_tri])

        return num_shapes