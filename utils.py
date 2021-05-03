import cv2
import numpy as np

MIN_CONTOUR_AREA = 5000
RECTANGLE_NUM_SIZE = 4


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def find_biggest_contour_approx(contours, min_contour_area=MIN_CONTOUR_AREA):
    biggest_contour_approx = np.array([])
    max_area = 0
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > min_contour_area:
            contour_perimeter = cv2.arcLength(contour, closed=True)
            contour_approx = cv2.approxPolyDP(contour, 0.02 * contour_perimeter, closed=True)
            if contour_area > max_area and len(contour_approx) == RECTANGLE_NUM_SIZE:
                max_area, biggest_contour_approx = contour_area, contour_approx
    return biggest_contour_approx, max_area


def reorder_contour_approx(contour):
    points = contour.reshape((4, 2))
    points_result = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)

    points_result[0] = points[np.argmin(add)]
    points_result[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    points_result[1] = points[np.argmin(diff)]
    points_result[2] = points[np.argmax(diff)]

    return points_result


def remove_shadow(img):
    dilated_img = cv2.dilate(img, np.ones((5, 5), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 83)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
    cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    _, thr_img = cv2.threshold(thr_img, 230, 255, cv2.THRESH_BINARY)
    return thr_img
