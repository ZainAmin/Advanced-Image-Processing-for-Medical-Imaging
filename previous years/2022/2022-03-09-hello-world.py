import numpy as np
import cv2 as cv

examples_folder = "../../project configuration/OpenCV 4 - C++/example_images"

img = cv.imread(examples_folder + "/lena.png", cv.IMREAD_UNCHANGED)
print(img.shape)

cv.imshow("Image", img)
cv.waitKey(0)
cv.destroyAllWindows()
