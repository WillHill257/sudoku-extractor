import sys

# add the model to the sys path so we can access it
sys.path.insert(1, "./MnistSimpleCNN-master/code/")

from copy import deepcopy
from skimage.io import imread, imshow, show
from skimage.color import rgb2gray
from skimage import img_as_float, img_as_ubyte, exposure
from skimage.filters import (
    threshold_local,
    threshold_li,
)
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology, measure
from scipy.ndimage import gaussian_filter, gaussian_laplace
from skimage.transform import resize
from os import mkdir
import torch
from models.modelM3 import ModelM3
from datasets import MnistDataset
from torchvision import transforms

# number of intensity levels in the image
L = 256

# function to display an image
def showImage(image):
    imshow(image, cmap="gray")
    show()


# calculate a representation of the sharpness of an image
def sharpnessScore(image):
    # get the gradients of the image in both directions
    gy, gx = np.gradient(image)

    # take the magnitude of the gradients
    gnorm = np.sqrt(gx ** 2 + gy ** 2)

    # the average magintude is our score
    return np.average(gnorm)


# find the contours (connected segments, with labels) in the image
def findContours(image):
    contours = measure.find_contours(image, 0.8)
    return contours


# plot the contours on top of an image (that they came from)
def plotContours(image, contours):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    plt.show()


# convert the contours to a bounding box representation
def findBoundingBox(contours):
    bounding_boxes = []

    for contour in contours:
        Xmin = int(np.floor(np.min(contour[:, 0])))
        Xmax = int(np.ceil(np.max(contour[:, 0])))
        Ymin = int(np.floor(np.min(contour[:, 1])))
        Ymax = int(np.ceil(np.max(contour[:, 1])))

        bounding_boxes.append([Xmin, Xmax, Ymin, Ymax])

    return bounding_boxes


# plot the bounding boxes over the image they came from and optionally include red/green box for whether a digit is present
def plotBoundingBoxes(boundingBoxes, imageShape, overlay=None, num_exists=None):
    for i, box in enumerate(boundingBoxes):
        r = [box[0], box[1], box[1], box[0], box[0]]
        c = [box[3], box[3], box[2], box[2], box[3]]
        if num_exists == None or not num_exists[i]:
            plt.plot(c, r, color="r")
        else:
            plt.plot(c, r, color="g")

    if np.any(overlay) != None:
        plt.imshow(overlay, cmap="gray")
    plt.show()


# determine the largest box (by area)
def findMaxBB(bounding_boxes):
    max = -1
    maxIndex = -1
    for i, bb in enumerate(bounding_boxes):
        area = (bb[1] - bb[0]) * (bb[3] - bb[2])
        if area > max:
            max = area
            maxIndex = i

    return bounding_boxes[maxIndex]


# determine which bounding boxes are squares (vertical and horizontal lines should be in large eneough proportion to each other)
def passSquares(boundingBoxes, toleranceRatio):
    # store the passed boxes
    passedBoxes = []

    # loop through the boxes
    for bb in boundingBoxes:
        # determine vertical and horizontal distances
        horizontalDistance = bb[1] - bb[0]
        verticalDistance = bb[3] - bb[2]

        # get the ratio of these two
        ratio = min(horizontalDistance, verticalDistance) / max(
            horizontalDistance, verticalDistance
        )

        # check if it passes - ratio approaches 1 if the distances are equal
        if ratio >= toleranceRatio:
            passedBoxes.append(bb)

    return passedBoxes


# find the horizontal and vertical lines of the sudoku grid
def findGridLines(image):
    # extract vertical and horizontal edges, respectively
    vert_SE = morphology.rectangle(41, 1)
    sudoku_v = morphology.binary_opening(image, vert_SE)

    hori_SE = morphology.rectangle(1, 41)
    sudoku_h = morphology.binary_opening(image, hori_SE)

    # get the vertical and horizontal grid lines together in a single image
    return sudoku_h + sudoku_v


# determines whether a cell contains a number
# threshold is the minumum percentage to pass as a number
def isDigitPresent(cropped_cell, threshold):

    num_ones = np.sum(cropped_cell == 1)
    p = np.sqrt(num_ones / len(cropped_cell.flatten()))

    return p >= threshold


# function to extract the sudoku grid
def extract_sudoku(filename):

    sudoku = None

    # read in the image, if it exists
    try:
        sudoku = imread(filename)
    except:
        print("Please choose a valid image")
        exit()

    # convert the image to greyscale
    sudoku = img_as_ubyte(rgb2gray(sudoku))

    # apply unmask filtering
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
    for i in range(1):
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

    # rescale the image so our later filters work with a single size (no further tuning required)
    sudoku_cropped = resize(sudoku_cropped, (450, 450))
    sharpness = sharpnessScore(sudoku_cropped)

    # fix up the intensity ditribution - makes thresholding easier
    sudoku_cropped = exposure.equalize_adapthist(sudoku_cropped)

    # this is used to crop out the numbers
    sudoku_cropped_copy = deepcopy(sudoku_cropped)

    # sharpen an already sharp image - enhance edges, makes easier to detect boxes and numbers without the noise (outside the main grid)
    sudoku_cropped = gaussian_laplace(sudoku_cropped, sigma=2)

    # apply a global threshold to the image
    threshold_value = threshold_li(sudoku_cropped)
    sudoku_cropped = sudoku_cropped >= threshold_value

    # extract the gridlines again
    sudoku_cropped = morphology.binary_dilation(sudoku_cropped, morphology.disk(2))
    sudoku_cropped_box = findGridLines(sudoku_cropped)

    # close up any holes in the gridlines
    sudoku_cropped_box = morphology.closing(
        sudoku_cropped_box, morphology.rectangle(4, 31)
    )
    sudoku_cropped_box = morphology.closing(
        sudoku_cropped_box, morphology.rectangle(31, 4)
    )

    # determine the bounding boxes
    contours = findContours(sudoku_cropped_box)
    bounding_boxes = findBoundingBox(contours)

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

    # filter the rectangles out
    passed_boxes = passSquares(bounding_boxes, 0.6)

    # if it fails, yolo
    if len(passed_boxes) < 81:
        print("Please take a more clear image, without any shading in the image")
        exit()

    # sort by the area of the box
    passed_boxes = sorted(
        passed_boxes, key=lambda box: (box[1] - box[0]) * (box[3] - box[2])
    )

    # find the median box area
    # find the median index
    medianIndex = len(passed_boxes) // 2
    numMediansToConsiderEitherSide = 1
    medianArea = 0
    for i in range(-numMediansToConsiderEitherSide, numMediansToConsiderEitherSide + 1):
        box = passed_boxes[medianIndex + i]
        medianArea += (box[1] - box[0]) * (box[3] - box[2])

    medianArea /= numMediansToConsiderEitherSide * 2 + 1

    # sort the boxes by the difference of their area to the median area
    passed_boxes = sorted(
        passed_boxes,
        key=lambda box: np.abs((box[1] - box[0]) * (box[3] - box[2]) - medianArea),
    )

    # take the 81 smallest
    sudoku_grid_boxes = passed_boxes[:81]

    """
    1) linearise the indices of the 81 boxes
    2) determine which grid boxes have digits in them
    3) crop out from pre-thresholded image and save the boxes with images for recognition (save with reference to its linear index)
    """

    # sort by y first
    sudoku_grid_boxes = sorted(sudoku_grid_boxes, key=lambda box: box[0])

    # sort by x
    for i in range(9):
        sudoku_grid_boxes[9 * i : (i + 1) * 9] = sorted(
            sudoku_grid_boxes[9 * i : (i + 1) * 9], key=lambda box: box[2]
        )

    # erode image to get rid of more noise
    # pick an SE based on the sharpness score
    SE = None
    if sharpness >= 0.03:
        SE = morphology.rectangle(5, 5)
    else:
        SE = morphology.rectangle(6, 6)

    sudoku_cropped_eroded = morphology.binary_erosion(sudoku_cropped, SE)

    # an array that stores whether a position contains a number
    num_exists = [False] * 81
    for i, bb in enumerate(sudoku_grid_boxes):

        # crop out each box (with interior padding so don't get any of the borders)
        border_padding = 5
        cropped_cell = sudoku_cropped_eroded[
            bb[0] + border_padding : bb[1] - border_padding,
            bb[2] + border_padding : bb[3] - border_padding,
        ]

        # the array that stores whether a digit is present at that location
        num_exists[i] = isDigitPresent(cropped_cell, 0.22)

    # create directory to save the cropped out number images
    dirname = filename[: filename.rfind(".")] + "_output/"
    try:
        mkdir(dirname)
    except:
        pass

    # store the cropped out images
    cropped_cells = []
    # the actual cropping out of the numbers
    for i in range(81):
        # only save the images that have a digit in them
        if num_exists[i]:
            bb = sudoku_grid_boxes[i]
            cropped_number = img_as_float(
                sudoku_cropped_copy[bb[0] : bb[1], bb[2] : bb[3]]
            )
            cropped_cells.append((i, cropped_number))
            plt.imsave(
                dirname + str(i) + ".png", cropped_number, format="png", cmap="gray"
            )

    # uncomment this to view the binary grid with red and green overlays for if a digit is present or not

    # plotBoundingBoxes(
    #     sudoku_grid_boxes,
    #     sudoku_cropped_eroded.shape,
    #     sudoku_cropped_eroded,
    #     num_exists,
    # )

    # return the cropped out cells with a digit in them
    return cropped_cells, num_exists


# format the cells as if they were MNIST digits
# make them 28x28
# threshold them to get better results
def format_cells(cropped_cells):
    output = []
    # loop through each cell
    for i, cell in cropped_cells:
        # resize
        cell_resized = resize(cell, (28, 28))
        threshold_value = threshold_li(cell_resized)
        cell_resized = cell_resized >= threshold_value

        # add to the output
        output.append((i, cell_resized))

    return output


def convert_to_dataloader(imageList):
    # create the dataloader, with no shuffling since order matters
    dataloader = torch.utils.data.DataLoader(imageList, shuffle=False, batch_size=1)

    return dataloader


# predict the digits from images
# use the pretrained model
def predict_digits(dataloader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # choose the model to use
    model1 = ModelM3().to(device)

    # load the trained model parameters
    model1.load_state_dict(
        torch.load(
            "MnistSimpleCNN-master/logs/modelM3/model000.pth",
            map_location=torch.device("cpu"),
        )
    )

    # do the prediction
    model1.eval()

    # no need to do gradient computations
    predictions = []
    with torch.no_grad():
        # loop through the dataloader
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)

            # run the input through the model
            output = model1(data)

            # determine the model's prediction
            pred = torch.topk(output, 2, dim=1).indices
            predictions.append(pred.detach().cpu().numpy().flatten())

    return predictions


# print the text-based sudoku grid
def printSudoku(digits, num_exists):
    digit_idx = 0

    # loop through each box
    for i in range(9):
        for j in range(9):

            # compute the linearised index of the box
            lin_idx = j + i * 9

            # if a digit exists in this box, print the number, otherwise print _
            if num_exists[lin_idx]:
                # use digit_idx to keep track of the digit to print
                # (since the array only stores the boxes with digits, not the boxes without)
                if digits[digit_idx][0] == 0:
                    print(digits[digit_idx][1], end=" ")
                else:
                    print(digits[digit_idx][0], end=" ")

                digit_idx += 1
            else:
                print("_", end=" ")
        print()


if __name__ == "__main__":
    # read in an image filename
    filename = input("Please enter the absolute path of the image to process: ").strip()

    # get the cropped cells with digits in them
    # this function will also save the digit images
    cropped_cells, num_exists = extract_sudoku(filename)

    # need to format the cells into the correct shapes
    cropped_cells = format_cells(cropped_cells)

    # prepare the model
    test_dataset = MnistDataset(training=False, transform=None)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False
    )

    # get the images
    images = [
        transforms.ToTensor()((1 - np.array(pair[1])).astype(np.float32))
        for pair in cropped_cells
    ]

    # create a dataloader
    dataloader = convert_to_dataloader(images)

    # use the trained MNIST model to predict the digits
    results = predict_digits(dataloader)

    # print the text-based sudoku puzzle
    printSudoku(results, num_exists)

    print()
    print(
        "The cropped boxes (with digits in them) can be found here:",
        filename[: filename.rfind(".")] + "_output/",
    )
