# import libraries
import numpy as np
import cv2 as cv2
from PIL import Image


# https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

# https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py
def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts

def disp_image(img):
    '''
    input: cv2 image 
    output: displays PIL image
    '''
    # You may need to convert the color.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    
    return im_pil

def read_shapes_text(text):
    with open(text, 'r') as fp:
        shape_count = fp.read().splitlines()
        assert(len(shape_count)==1)
        shape_count = shape_count[0].strip().split(', ')
        shape_count[0] = int(shape_count[0].strip().split(':')[1])
        shape_count[1] = int(shape_count[1].strip().split(':')[1])
        shape_count[2] = int(shape_count[2].strip().split(':')[1])
        shape_count = np.array(shape_count).reshape(1,-1)
    return shape_count

def accuracy(y, y_hat):
    num_images = y.shape[0]
    difference = y - y_hat
    
    num_squares = abs(difference[:,0:1]).sum()
    num_circles = abs(difference[:,1:2]).sum()
    num_triangles = abs(difference[:,2:3]).sum()
    num_shapes = difference.sum()
    
    acc_squares = (num_images - abs(num_squares)) / num_images
    acc_circles = (num_images - abs(num_circles)) / num_images
    acc_triangles = (num_images - abs(num_triangles)) / num_images
    acc_shapes = (num_images - abs(num_shapes)) / num_images
    
    print((acc_squares, acc_circles, acc_triangles), acc_shapes)
    
    return (acc_squares, acc_circles, acc_triangles), acc_shapes, difference
