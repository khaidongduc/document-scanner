import cv2
import numpy as np
import utils

#
IMG_PATH = "sample.png"
IMG_HEIGHT = 640
IMG_WIDTH = 480

CANNY_SIGMA = 0.33
KERNEL = np.ones((5, 5))
NUM_SHRINKING_PIXEL = 5

# image processing
img = cv2.imread(IMG_PATH)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray scale
imgBlur = cv2.GaussianBlur(imgGray, (0, 0), 1)
imgEdged = utils.auto_canny(imgBlur, CANNY_SIGMA)
imgDilated = cv2.dilate(imgEdged, kernel=KERNEL, iterations=2)
imgEroded = cv2.erode(imgDilated, kernel=KERNEL, iterations=1)

# finding the biggest contour approx
contours, hierarchy = cv2.findContours(imgEroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
biggestContourApprox, maxArea = utils.find_biggest_contour_approx(contours)
biggestContourApprox = utils.reorder_contour_approx(biggestContourApprox)

imgContours = img.copy()
imgBiggestContour = img.copy()
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
cv2.drawContours(imgBiggestContour, biggestContourApprox, -1, (0, 255, 0), 10)

# warping image
contour_perspective = np.float32(biggestContourApprox)
warped_perspective = np.float32([[0, 0], [IMG_WIDTH, 0],
                                 [0, IMG_HEIGHT], [IMG_WIDTH, IMG_HEIGHT]])
transform_matrix = cv2.getPerspectiveTransform(contour_perspective, warped_perspective)
imgWarped = cv2.warpPerspective(img, transform_matrix, (IMG_WIDTH, IMG_HEIGHT))

# correction
imgWarped = imgWarped[NUM_SHRINKING_PIXEL:imgWarped.shape[0] - NUM_SHRINKING_PIXEL,
                      NUM_SHRINKING_PIXEL:imgWarped.shape[1] - NUM_SHRINKING_PIXEL]

# adaptive
imgWarpedGray = cv2.cvtColor(imgWarped, cv2.COLOR_BGR2GRAY)
imgAdaptive = cv2.adaptiveThreshold(imgWarpedGray, 255, 1, 1, 7, 2)
imgAdaptive = cv2.bitwise_not(imgAdaptive)
imgAdaptive = cv2.medianBlur(imgAdaptive, 3)


cv2.imshow("image", img)
cv2.imshow("image edged", imgEdged)
cv2.imshow("image biggest contour", imgBiggestContour)
cv2.imshow("image warped", imgWarped)
cv2.imshow("image adaptive", imgAdaptive)

cv2.waitKey(0)
cv2.destroyAllWindows()
