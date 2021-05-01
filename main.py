from document_scanner import DocumentScanner
import os

import cv2

document_scanner = DocumentScanner()

sample_folder = "sample"
result_folder = "result"

file_name = "sample.png"
sample_img_path = os.path.join(sample_folder, file_name)
original_img = cv2.imread(sample_img_path)
img = document_scanner.scan_document(original_img)


cv2.imshow("res", img)
cv2.waitKey(0)

result_img_path = os.path.join(result_folder, file_name)
cv2.imwrite(result_img_path, img)
