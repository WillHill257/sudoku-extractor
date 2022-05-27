from copy import deepcopy
from skimage.io import imread, imsave, imshow, show
from skimage.color import rgb2gray
from skimage import img_as_float, img_as_ubyte, exposure
from skimage.filters import (
    threshold_local,
    threshold_li,
)
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology, measure
from scipy.ndimage import gaussian_filter, convolve, gaussian_laplace
from skimage.draw import polygon_perimeter


"""
1) Convert image to greyscale
2) Convert image to binary image
3) Box detection - find vertical and horizontal lines, then find contours
4) Extract the main sudoku grid
5) Box detection again
"""

L = 256


def showImage(image):
    imshow(image, cmap="gray")
    show()


def findContours(image):
    contours = measure.find_contours(image, 0.8)
    return contours


def findBoundingBox(contours, display=False):
    bounding_boxes = []

    for contour in contours:
        Xmin = int(np.floor(np.min(contour[:, 0])))
        Xmax = int(np.ceil(np.max(contour[:, 0])))
        Ymin = int(np.floor(np.min(contour[:, 1])))
        Ymax = int(np.ceil(np.max(contour[:, 1])))

        bounding_boxes.append([Xmin, Xmax, Ymin, Ymax])

    if display:
        with_boxes = np.zeros(sudoku_box.shape)

        for box in bounding_boxes:
            # [Xmin, Xmax, Ymin, Ymax]
            r = [box[0], box[1], box[1], box[0], box[0]]
            c = [box[3], box[3], box[2], box[2], box[3]]
            rr, cc = polygon_perimeter(r, c, with_boxes.shape)
            with_boxes[rr, cc] = 1  # set color white

        plt.imshow(with_boxes, cmap="gray")
        plt.show()

    return bounding_boxes


def findMaxBB(bounding_boxes):
    max = -1
    maxIndex = -1
    for i, bb in enumerate(bounding_boxes):
        area = (bb[1] - bb[0]) * (bb[3] - bb[2])
        if area > max:
            max = area
            maxIndex = i

    return bounding_boxes[maxIndex]


def findGridLines(image):
    # extract vertical and horizontal edges, respectively
    vert_SE = morphology.rectangle(31, 1)
    sudoku_v = morphology.binary_opening(image, vert_SE)

    hori_SE = morphology.rectangle(1, 31)
    sudoku_h = morphology.binary_opening(image, hori_SE)

    # get the vertical and horizontal grid lines together in a single image
    return sudoku_h + sudoku_v


if __name__ == "__main__":
    # read in an image filename
    # filename = input("Please enter the absolute path of the image to process: ").strip()
    filename = "./v2_test/image50.jpg"
    # filename = "./v2_test/image18.jpg"
    # filename = "/Users/williamhill/Downloads/v2_test/image46.jpg"

    # read in the image
    sudoku = imread(filename)

    # convert the image to greyscale
    sudoku = img_as_ubyte(rgb2gray(sudoku))

    # apply highboost filtering
    sudoku = gaussian_filter(sudoku, 1)
    blurredImage = gaussian_filter(sudoku, 1.5)
    mask = sudoku - blurredImage
    sudoku = sudoku + 1 * mask

    # this a copy of the sharpened image
    sudoku_copy = np.copy(sudoku)

    # apply thresholding in a neighbourhood
    local_thresh = threshold_local(sudoku, 151)
    sudoku = sudoku < local_thresh

    # perform morphological operations to help remove noise
    SE = morphology.rectangle(3, 3)
    sudoku = morphology.binary_opening(sudoku, SE)
    sudoku = morphology.binary_erosion(sudoku, morphology.disk(2))

    # do the big box extraction twice to get rid of extra shadows (like the piece of paper the grid is on)
    # doing it twice doesn't affect if the main grid extracted in the first pass
    sudoku_cropped = deepcopy(sudoku_copy)
    for i in range(2):
        # extract the gridlines
        sudoku_box = findGridLines(sudoku)

        # find contours, use them to determine the bounding boxes
        # cropping out the largest bounding box, i.e. the sudoku itself
        contours = findContours(sudoku_box)
        bounding_boxes = findBoundingBox(contours)
        max_bb = findMaxBB(bounding_boxes)

        # use the max bounding box to extract the sudoku grid itself (for the binary image) - for further cropping
        sudoku = sudoku[max_bb[0] : max_bb[1], max_bb[2] : max_bb[3]]

        # crop for the original image as well
        sudoku_cropped = sudoku_cropped[max_bb[0] : max_bb[1], max_bb[2] : max_bb[3]]

    # fix up the intensity ditribution - makes thresholding easier
    sudoku_cropped = exposure.equalize_adapthist(sudoku_cropped)

    # sharpen an already sharp image - enhance edges, makes easier to detect boxes and numbers without the noise (outside the main grid)
    sudoku_cropped = gaussian_laplace(sudoku_cropped, sigma=1)

    # apply a global threshold to the image
    threshold_value = threshold_li(sudoku_cropped)
    sudoku_cropped = sudoku_cropped >= threshold_value

    # extract the gridlines again
    sudoku_cropped_box = findGridLines(sudoku_cropped)

    # close up any holes in the gridlines
    sudoku_cropped_box = morphology.closing(
        sudoku_cropped_box, morphology.rectangle(1, 31)
    )
    sudoku_cropped_box = morphology.closing(
        sudoku_cropped_box, morphology.rectangle(31, 1)
    )

    # determine the bounding boxes
    contours = findContours(sudoku_cropped_box)
    bounding_boxes = findBoundingBox(contours, True)
    max_bb = findMaxBB(bounding_boxes)

    # find the 81 boxes to crop out
    """
    1) filter by the absolute value of the difference between the horizontal and vertical edges (can average these for each box) - set tolerance
    2) the rectangles will now be excluded from the list
    3) now sort by the area of the box
    4) now we have the smallest squares, then the sudoku squares, then large squares
    5) take the median of the list as the "true" area
    6) sort by abosluted difference with to this area
    7) take the 81 smallest
    """
