# Stitch two graffiti images manually

import cv2 as cv
import numpy as np

img1pts = np.zeros((4, 2), dtype="float")
img2pts = np.zeros((4, 2), dtype="float")
i = 0

# This method is used to get points
def get_points(event, x, y, flags, params):
    global i
    pts = params[0]
    img = params[1]
    if event == cv.EVENT_LBUTTONDOWN:
        pts[i, 0], pts[i, 1] = x, y
        i = i + 1
        cv.circle(img, (x, y), 1, (0, 255, 0), 2)
        hw = 10
        cv.rectangle(img, (x - hw, y - hw), (x + hw, y + hw), (0, 0, 255), 2)


img1 = cv.imread('images/img3.ppm')
cv.namedWindow("Image1", cv.WINDOW_AUTOSIZE)
params = [img1pts, img1]
cv.setMouseCallback("Image1", get_points, params)

while 1:
    cv.imshow("Image1", img1)
    k = cv.waitKey(1) & 0xFF
    print(i)
    if k == 27:
        break
    if i >= 4:
        break

img2 = cv.imread('images/img2.ppm')
cv.namedWindow("Image2", cv.WINDOW_AUTOSIZE)
params = [img2pts, img2]
cv.setMouseCallback("Image2", get_points, params)
i = 0


while 1:
    cv.imshow("Image2", img2)
    k = cv.waitKey(1) & 0xFF
    print(i)
    if k == 27:
        break
    if i >= 4:
        break

# Find homography transformation
(H, _) = cv.findHomography(img1pts, img2pts)

# Find warped image
result = cv.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
# Add image 2
result[0:img2.shape[0], 0:img2.shape[1]] = img2
cv.imwrite('images/img4.ppm', result)
cv.namedWindow("Stitched", cv.WINDOW_AUTOSIZE)
cv.imshow("Stitched", result)

cv.waitKey(1)
cv.destroyAllWindows()
