from skimage.io import imread, imsave, imshow, show
from skimage.color import rgb2gray
from skimage import img_as_float, img_as_ubyte
from skimage.filters import threshold_local
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from scipy.ndimage import gaussian_filter

"""
1) Convert image to greyscale
2) Convert image to binary image
3) Noise removal
4) Box detection - find vertical and horizontal lines, then find contours
"""

L = 256


def showImage(image):
    imshow(image, cmap="gray")
    show()


if __name__ == "__main__":
    # read in an image filename
    filename = input("Please enter the absolute path of the image to process: ").strip()
    # filename = "/Users/williamhill/Downloads/v2_test/image18.jpg"
    # filename = "/Users/williamhill/Downloads/v2_test/image46.jpg"

    # read in the image
    sudoku = imread(filename)

    # convert the image to greyscale
    sudoku = img_as_ubyte(rgb2gray(sudoku))

    # apply highboost filtering
    sudoku = gaussian_filter(sudoku, 1)
    blurredImage = gaussian_filter(sudoku, 1.5)
    mask = sudoku - blurredImage
    sudoku = sudoku + 5 * mask

    # apply thresholding in a neighbourhood
    local_thresh = threshold_local(sudoku, 151)
    sudoku = sudoku < local_thresh

    # perform morphological operations to help remove noise and "fill" the numbers in
    SE = morphology.rectangle(3, 3)
    sudoku = morphology.binary_opening(sudoku, SE)
    sudoku = morphology.binary_erosion(sudoku, SE)

    # and, voila
    showImage(sudoku)
