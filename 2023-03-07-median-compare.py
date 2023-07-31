import numpy as np
import cv2 as cv
import time

examples_folder = "../project configuration/example_images"

img = cv.imread(examples_folder + "/rice.png", cv.IMREAD_GRAYSCALE)
height = img.shape[0]
width = img.shape[1]
out = np.zeros((height, width, 1), np.uint8)

k = 7
h = int(k / 2)

t0 = time.time()
for y in range(h, height-h):
    for x in range(h, width-h):
        window = img[y - h:y + h + 1, x - h:x + h + 1]
        out[y][x] = np.median(window)
print(f"Processing time (custom median) = {time.time()-t0:.3f} seconds")

t0 = time.time()
out2 = cv.medianBlur(img, k)
print(f"Processing time (OpenCV median) = {time.time()-t0:.3f} seconds")

cv.imshow("Image median-filtered (custom)", out)
cv.imshow("Image median-filtered (OpenCV)", out2)

cv.waitKey(0)
cv.destroyAllWindows()
