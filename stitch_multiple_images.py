# SIFT feature detection

import cv2 as cv
import numpy as np
import imutils

# Read two images
imageA = cv.imread('images/img5.ppm')
imageB = cv.imread('images/img2.ppm')
imageC = cv.imread('images/img3.ppm')


# Created a method to stitch two images
def stitch(image1, image2):
    # Resize
    image1 = imutils.resize(image1, height=400)
    image2 = imutils.resize(image2, height=400)

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
    return result


# Call the stitch method iteratively
result = stitch(imageA, imageB)
result = stitch(result, imageC)
cv.imshow("Stitched", result)
cv.imwrite('images/img6.ppm', result)
cv.imshow("Image A", imageA)
cv.imshow("Image B", imageB)

cv.waitKey(0)
cv.destroyAllWindows()
