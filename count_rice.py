import cv2 as cv
import numpy as np

# Read image
img_ori = cv.imread('images/rice.png', cv.IMREAD_COLOR)
cv.namedWindow('Image_original', cv.WINDOW_AUTOSIZE)
cv.imshow('Image_original', img_ori)

# Convert to gray
img = cv.cvtColor(img_ori, cv.COLOR_BGR2GRAY)
# Sharpen the image
im = cv.GaussianBlur(img, (11, 11), 0)
hp = cv.subtract(img, im)
alpha = 1
img = cv.add(img, alpha*hp)

# Median filtering
kernel_size = 5
img = cv.medianBlur(img, kernel_size)

# Get the background and subtract it
kernel = np.ones((25, 25), np.uint8)
background = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

diff = cv.subtract(img, background)

# Apply a threshold
ret, threshold = cv.threshold(diff, 70, 255, cv.THRESH_BINARY)

# Count contours
im2, contours, hierarchy = cv.findContours(threshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))

cv.drawContours(img_ori, contours, -1, (0, 0, 255), 3)

cv.namedWindow('Image_filtered', cv.WINDOW_AUTOSIZE)
cv.imshow('Image_filtered', img)

cv.namedWindow('Image_diff', cv.WINDOW_AUTOSIZE)
cv.imshow('Image_diff', diff)

cv.namedWindow('Image_threshold', cv.WINDOW_AUTOSIZE)
cv.imshow('Image_threshold', threshold)

cv.namedWindow('Image_result', cv.WINDOW_AUTOSIZE)
cv.imshow('Image_result', img_ori)
cv.waitKey(0)
cv.destroyAllWindows()


