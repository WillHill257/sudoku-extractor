from skimage.io import imread, imsave, imshow, show
from skimage.color import rgb2gray
from skimage import img_as_float, img_as_ubyte, exposure
from skimage.filters import threshold_local, try_all_threshold
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology, measure
from scipy.ndimage import gaussian_filter, convolve
from skimage.draw import polygon_perimeter


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


def findContours(image):
    contours = measure.find_contours(image, 0.8)
    return contours


def findBoundingBox(contours):
    bounding_boxes = []

    for contour in contours:
        Xmin = int(np.floor(np.min(contour[:, 0])))
        Xmax = int(np.ceil(np.max(contour[:, 0])))
        Ymin = int(np.floor(np.min(contour[:, 1])))
        Ymax = int(np.ceil(np.max(contour[:, 1])))

        bounding_boxes.append([Xmin, Xmax, Ymin, Ymax])

    # with_boxes = np.zeros(sudoku_box.shape)

    # for box in bounding_boxes:
    #     #[Xmin, Xmax, Ymin, Ymax]
    #     r = [box[0], box[1], box[1], box[0], box[0]]
    #     c = [box[3], box[3], box[2], box[2], box[3]]
    #     rr, cc = polygon_perimeter(r, c, with_boxes.shape)
    #     with_boxes[rr, cc] = 1  # set color white

    # plt.imshow(with_boxes, cmap="gray")
    # plt.show()
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


if __name__ == "__main__":
    # read in an image filename
    # filename = input("Please enter the absolute path of the image to process: ").strip()
    filename = "./v2_test/image205.jpg"
    # filename = "./v2_test/image46.jpg"
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
    sudoku_copy = np.copy(sudoku)

    # apply thresholding in a neighbourhood
    local_thresh = threshold_local(sudoku, 151)
    sudoku = sudoku < local_thresh

    # perform morphological operations to help remove noise and "fill" the numbers in
    SE = morphology.rectangle(3, 3)
    sudoku = morphology.binary_opening(sudoku, SE)
    # SE = morphology.disk(1)
    # sudoku = morphology.binary_closing(sudoku, SE)

    vert_SE = morphology.rectangle(25, 1)
    sudoku_v = morphology.binary_opening(sudoku, vert_SE)

    hori_SE = morphology.rectangle(1, 25)
    sudoku_h = morphology.binary_opening(sudoku, hori_SE)

    # SE = morphology.disk(3)
    sudoku_box = sudoku_h + sudoku_v
    # sudoku_box = morphology.binary_closing(sudoku_box, SE)

    # and, voila
    # showImage(sudoku)

    # cropping out the largest bounding box, i.e. the sudoku itself
    contours = findContours(sudoku_box)
    bounding_boxes = findBoundingBox(contours)
    max_bb = findMaxBB(bounding_boxes)

    sudoku_cropped = sudoku_copy[max_bb[0]:max_bb[1], max_bb[2]:max_bb[3]]
    showImage(sudoku_cropped)

    sudoku_cropped = exposure.equalize_adapthist(sudoku_cropped)
    # laplacian = np.array([
    #     [-1, -1, -1],
    #     [-1, 9, -1],
    #     [-1, -1, -1]
    # ])
    showImage(sudoku_cropped)

    # sudoku_cropped = convolve(sudoku_cropped, laplacian)
    # showImage(sudoku_cropped)

    fig, ax = try_all_threshold(sudoku_cropped, figsize=(10, 6), verbose=False)
    plt.show()
    # apply thresholding in a neighbourhood
    local_thresh = threshold_local(sudoku_cropped, 251)
    sudoku_cropped = sudoku_cropped < local_thresh
    showImage(sudoku_cropped)


# showImage(morphology.binary_closing(sudoku_h + sudoku_v, SE))
