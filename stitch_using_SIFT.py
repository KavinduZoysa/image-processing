# SIFT feature detection

import cv2 as cv
import numpy as np
import imutils

# Read two images
image1 = cv.imread('images/img3.ppm')
image2 = cv.imread('images/img2.ppm')

# Resize
image1 = imutils.resize(image1, width=400)
image2 = imutils.resize(image2, width=400)

# Find gray images
grayA = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
grayB = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

# Find SIFT features
sift = cv.xfeatures2d.SIFT_create()
(kpA, desA) = sift.detectAndCompute(grayA, None)
(kpB, desB) = sift.detectAndCompute(grayB, None)

# Convert points to np array
kpA = np.float32([kp.pt for kp in kpA])
kpB = np.float32([kp.pt for kp in kpB])

# Matches each points
matcher = cv.DescriptorMatcher_create("BruteForce")
rawMatches = matcher.knnMatch(desA, desB, 2)
matches = []
H = np.ones((3, 3), dtype=float)

# loop over the raw matches
for m in rawMatches:
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        matches.append((m[0].trainIdx, m[0].queryIdx))

# computing a homography (need at least 4 matches)
if len(matches) > 4:
    # construct the two sets of points
    ptsA = np.float32([kpA[i] for (_, i) in matches])
    ptsB = np.float32([kpB[i] for (i, _) in matches])
    # compute the homography using RANSAC method
    (H, status) = cv.findHomography(ptsA, ptsB, cv.RANSAC, 4.0)

result = cv.warpPerspective(image1, H, (image1.shape[1] + image2.shape[1], image1.shape[0]))
result[0:image2.shape[0], 0:image2.shape[1]] = image2

(hA, wA) = image1.shape[:2]
(hB, wB) = image2.shape[:2]
vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
vis[0:hA, 0:wA] = image1
vis[0:hB, wA:] = image2

# loop over the matches
for ((trainIdx, queryIdx), s) in zip(matches, status):
    # only process the match if the keypoint was successfully
    # matched
    if s == 1:
        # draw the match
        ptA = (int(kpA[queryIdx][0]), int(kpA[queryIdx][1]))
        ptB = (int(kpB[trainIdx][0]) + wA, int(kpB[trainIdx][1]))
        cv.line(vis, ptA, ptB, (0, 255, 0), 1)

cv.imshow("Stitched", result)
cv.imshow("Mapping", vis)
cv.imshow("Image A", image1)
cv.imshow("Image B", image2)

cv.waitKey(0)
cv.destroyAllWindows()
