# import installed libraries
import numpy as np
import cv2 as cv2
from sklearn.cluster import MiniBatchKMeans

def reduce_C_space(image):
    '''
    input: image shape (500, 500, 3)
    output: image shape (500,500,3) with exactly two unique RGB pixels
    '''
    assert(image.shape == (500, 500, 3))
    
    # load the image and grab its width and height
    (h, w) = image.shape[:2]

    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = 2)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    #image = image.reshape((h, w, 3))

    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    #image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    
    # number of unique pixels
    unique_pix = np.unique(quant.reshape(-1, quant.shape[2]), axis=0)
    
    #assert(unique_pix.shape[0]==2)
    assert(quant.shape == (500, 500, 3))
    
    return quant

def convert_bw(image):
    '''
    input: image shape (500, 500, 3)
    output: B&W image shape (500,500,3)
    '''
    # input assertations
    assert(image.shape == (500, 500, 3))

    # number of unique pixels
    unique_pix = np.unique(image.reshape(-1, image.shape[2]), axis=0)
    color_1 = unique_pix[0]
    
    # encase image is blank i.e. only bacground colour
    try:
        color_2 = unique_pix[1]
    except:
        color_2 = unique_pix[0]

    # Get indicies of two colors
    indices_1 = np.where(np.all(image == color_1, axis=-1))
    indices_2 = np.where(np.all(image == color_2, axis=-1))
    
    # number of pixels represented by each color
    pixels_color_1 = image[indices_1].shape[0]
    pixels_color_2 = image[indices_2].shape[0]

    # set background color to color represented by max pixels
    if pixels_color_1 > pixels_color_2:
        background_color = color_1
        shape_color = color_2
    else:
        background_color = color_2
        shape_color = color_1

    # Get indicies of background pixels
    indices_list = np.where(np.all(image == background_color, axis=-1))

    # Swap background color to black
    image[indices_list] = [0,0,0]

    # Get indicies of shape pixels
    indices_list = np.where(np.all(image == shape_color, axis=-1))

    # Swap background color to white
    image[indices_list] = [255,255,255]
    
    # output assertations
    assert(image.shape == (500, 500, 3))
    
    return image

def deionize_img(image, blur=10):
    '''
    input: image shape (500, 500, 3)
    output: denoized image shape (500,500,3)
    '''
    # input assertations
    assert(image.shape == (500, 500, 3))
    
    image = cv2.fastNlMeansDenoisingColored(image,None,blur,10,7,21)
    
    # output assertations
    assert(image.shape == (500, 500, 3))
    return image

def preprocess(image, blur=100):

    image = deionize_img(convert_bw(reduce_C_space(image)), blur=blur)
    
    return image