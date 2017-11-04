import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

DATA_PATH = Path('../train_screenshots').resolve()
FILE_FORMAT = 'screenshot{}.jpg'
IMAGE_NUM_START = 1
IMAGE_NUM_STOP = 9001

"""macOS buttons are colorful on the active window"""
template = cv2.imread('../buttons.jpg', 1)
templateOffsetX = -3
templateOffsetY = -2

xMinOffset = 65
yMinOffset = 50

# w, h = template.shape[1::-1]


"""Get labels for training data"""
with open('../train_coords.txt', 'r') as f:
    labels = [tuple(tuple(int(val) for val in coord.split(',')) for coord in line.strip().split('|')) for line in f.read().strip().split('\n')]

# print(labels)


def getTopLeft(image):
    """
    Get the top left corner's coordinate of the active window in the image
    :param image: image to be searched
    :return: top left corner (x, y)
    """
    # Apply template Matching
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
    _, _, _, maxLoc = cv2.minMaxLoc(result)

    return maxLoc[0] + templateOffsetX, maxLoc[1] + templateOffsetY


def autoCanny(image, sigma=0.33):
    """
    Do canny edge detection with automagically tuned parameters (based on the image)
    :param image: image to edge detect
    :param sigma: amount by which to vary the parameters
    :return: boolean image of edges
    """
    # apply automatic Canny edge detection using the computed median
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return cv2.Canny(image, lower, upper)

def getBottomRight(imageEdges: np.array):
    """
    Get the x coordinate of the right edge and the y coordinate of the bottom edge of the active window. Attempt
    until a reasonable bottomRight corner is found.

    :param imageEdges: the image edges, with the active window in the top left
    :return: the x coordinate of the right edge of the active window
    """

    # First find the right edge, which we assume we get correct every time

    # Go right until an edge is reached
    xRight = xMinOffset
    bestX = 0
    minXNum = 12 * 255

    while xRight < imageEdges.shape[1]:
        if not sum(imageEdges[0:5, xRight]):
            # Improve the guess at best x coord by searching for a good vertical edge.
            # Also check that there is enough of a vertical edge
            maxNum = 0
            bestX = 0
            for possibleX in range(xRight - 4, xRight + 6):
                try:
                    newNum = sum(imageEdges[0:25, possibleX])
                    if newNum > maxNum and newNum > minXNum:
                        maxNum = newNum
                        bestX = possibleX
                except Exception as e:
                    pass
                    # print(e)

            if bestX:
                break
        xRight += 1

    if bestX == 0:
        bestX = imageEdges.shape[1] - 1

    # Find the y value
    yBottom = yMinOffset
    bestY = 0
    minYNum = (bestX * 255) * 1 // 2

    while yBottom < imageEdges.shape[0]:
        if sum(imageEdges[yBottom, 0:5]) != 0:
            maxNum = 0
            bestY = 0
            for possibleY in range(yBottom - 4, yBottom + 6):
                try:
                    newNum = sum(imageEdges[possibleY, 0:bestX])
                    if newNum > maxNum and newNum > minYNum:
                        maxNum = newNum
                        bestY = possibleY
                except Exception as e:
                    pass
                    # print(e)

            if bestY:
                break
        yBottom += 1

    if bestY == 0:
        bestY = imageEdges.shape[0] - 1


    return bestX, bestY


def findActiveWindow():
    """
    Finds the active window in each image
    """

    # For each image
    for imgNum in range(IMAGE_NUM_START, IMAGE_NUM_STOP):
        if imgNum % 100 == 0:
            print('Processed {} images'.format(imgNum))

        imgPath = DATA_PATH / Path(FILE_FORMAT.format(imgNum))

        imgOrig = cv2.imread(str(imgPath), 1)
        # print(imgOrig.shape, template.shape)

        # Get and check the top left
        topLeft = getTopLeft(imgOrig.copy())
        # All top left labels are perfect :3

        imgCropped = imgOrig[topLeft[1]:, topLeft[0]:, :]

        edges = autoCanny(imgCropped)
        # print('Edges: {}'.format(edges.shape))

        guesses  = getBottomRight(edges)
        rightGuess = guesses[0] + topLeft[0]
        bottomGuess = guesses[1] + topLeft[1]
        # print('Found {}, actual {}'.format(guesses, (labels[imgNum - 1][1][0] - topLeft[0], labels[imgNum - 1][1][1] - topLeft[1])))

        if abs(rightGuess - labels[imgNum - 1][1][0]) > 2:
            print('Found {}, actual {}'.format((rightGuess, bottomGuess), labels[imgNum - 1][1]))

        # if abs(rightGuess - labels[imgNum - 1][1][0]) + abs(bottomGuess - labels[imgNum - 1][1][1]) > 5:
        #     print('Found {}, actual {}'.format((rightGuess, bottomGuess), labels[imgNum - 1][1]))

        # plt.subplot(121), plt.imshow(imgCropped, cmap='gray')
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122), plt.imshow(edges, cmap='gray')
        # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        # plt.show()

if __name__ == '__main__':
    findActiveWindow()

