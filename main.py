from document_scanner import DocumentScanner
import os

import cv2

document_scanner = DocumentScanner()

sample_folder = "sample"
result_folder = "result"

file_name = "sample7.jpg"
sample_img_path = os.path.join(sample_folder, file_name)
original_img = cv2.imread(sample_img_path)
img = document_scanner.scan_document(original_img)


imgShow = cv2.resize(img, (480, 640))
cv2.imshow("res", imgShow)
cv2.waitKey(0)

result_img_path = os.path.join(result_folder, file_name)
cv2.imwrite(result_img_path, img)
