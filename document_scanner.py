import cv2
import numpy as np
import utils

IMG_WIDTH = 480
IMG_HEIGHT = 640

CANNY_SIGMA = 0.33

DILATE_KERNEL = np.ones((3, 3))
ERODE_KERNEL = np.ones((3, 3))

GAUSSIAN_BLUR_KERNEL_SIZE = (1, 1)
GAUSSIAN_BLUR_BORDER_TYPE = cv2.BORDER_DEFAULT

CONTOUR_MODE = cv2.REDUCE_SUM
CONTOUR_METHOD = cv2.CHAIN_APPROX_SIMPLE

NUM_SHRINKING_PIXEL = 5


class DocumentScanner:
    def __init__(self,
                 img_size=(IMG_WIDTH, IMG_HEIGHT),
                 canny_sigma=CANNY_SIGMA,
                 dilate_kernel=DILATE_KERNEL,
                 erode_kernel=ERODE_KERNEL,
                 gaussian_blur_kernel_size=GAUSSIAN_BLUR_KERNEL_SIZE,
                 gaussian_blur_border_type=GAUSSIAN_BLUR_BORDER_TYPE,
                 contour_mode=CONTOUR_MODE,
                 contour_method=CONTOUR_METHOD,
                 min_contour_area=utils.MIN_CONTOUR_AREA,
                 num_shrinking_pixel=NUM_SHRINKING_PIXEL):
        self.__img_width, self.__img_height = img_size
        self.__canny_sigma = canny_sigma
        self.__dilate_kernel = dilate_kernel
        self.__erode_kernel = erode_kernel
        self.__gaussian_blur_kernel_size = gaussian_blur_kernel_size
        self.__gaussian_blur_border_type = gaussian_blur_border_type
        self.__contour_mode = contour_mode
        self.__contour_method = contour_method
        self.__min_contour_area = min_contour_area
        self.__num_shrinking_pixel = num_shrinking_pixel

    def scan_document(self, img_source):
        img = cv2.imread(img_source)
        imgResized = cv2.resize(img, (self.__img_width, self.__img_height))
        imgProcessed = self.__process_img(imgResized)
        biggest_contour_approx = self.__find_biggest_contour(imgProcessed, img)
        imgWarped = self.__warp_image(imgResized, biggest_contour_approx)
        return imgWarped

    def __process_img(self, img):
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray scale
        imgBlur = cv2.GaussianBlur(imgGray, self.__gaussian_blur_kernel_size, self.__gaussian_blur_border_type)
        imgEdged = utils.auto_canny(imgBlur, self.__canny_sigma)
        imgDilated = cv2.dilate(imgEdged, kernel=self.__dilate_kernel, iterations=2)
        imgEroded = cv2.erode(imgDilated, kernel=self.__erode_kernel, iterations=1)
        return imgEroded

    def __find_biggest_contour(self, img, img_org):
        try:
            contours, hierarchy = cv2.findContours(img, self.__contour_mode, self.__contour_method)
            biggestContourApprox, maxArea = utils.find_biggest_contour_approx(contours, self. __min_contour_area)
            biggestContourApprox = utils.reorder_contour_approx(biggestContourApprox)
            return biggestContourApprox
        except ValueError as value_error:
            raise ValueError("Unable to find contour find current settings")

    def __warp_image(self, img, contour_approx):
        contour_perspective = np.float32(contour_approx)
        warped_perspective = np.float32([[0, 0], [self.__img_width, 0],
                                         [0, self.__img_height], [self.__img_width, self.__img_height]])
        transform_matrix = cv2.getPerspectiveTransform(contour_perspective, warped_perspective)
        imgWarped = cv2.warpPerspective(img, transform_matrix, (IMG_WIDTH, IMG_HEIGHT))
        # correction
        imgWarped = imgWarped[self.__num_shrinking_pixel:imgWarped.shape[0] - self.__num_shrinking_pixel,
                              self.__num_shrinking_pixel:imgWarped.shape[1] - self.__num_shrinking_pixel]
        return imgWarped
