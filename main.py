from document_scanner import DocumentScanner
import os
from cv2 import imshow, waitKey, imwrite

document_scanner = DocumentScanner()

sample_folder = "sample"
result_folder = "result"

file_name = "sample.png"
sample_img_path = os.path.join(sample_folder, file_name)
img = document_scanner.scan_document(sample_img_path)

imshow("Result", img)
waitKey(0)

result_img_path = os.path.join(result_folder, file_name)
imwrite(result_img_path, img)
