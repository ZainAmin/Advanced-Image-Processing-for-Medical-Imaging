import numpy as np
import cv2 as cv
import time

img = np.zeros((512, 512), np.uint8)
print(img.shape)

t0 = time.time()
for y in range(0, img.shape[0]):
    for x in range(0, img.shape[1]):
        img[y][x] = (y / (img.shape[0]-1))*255
print(f"Elapsed time = {time.time()-t0:.3f}")


cv.imshow("Image", img)
cv.waitKey(0)
cv.destroyAllWindows()
