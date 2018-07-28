import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('images/justice.jpg', cv.IMREAD_GRAYSCALE)

brightness = 200
# just adding
imgb = img + brightness

# adding pixel by pixel
h = img.shape[0]
w = img.shape[1]
for i in range(0, h):
    for j in range(0, w):
        imgb[i, j] = min(img[i, j] + brightness, 255)

imgc = cv.add(img, brightness)

f, axarr = plt.subplots(1, 3)

index1 = 500
index2 = 700

print(img[index1, index2])
print(img[index1, index2] + brightness)
print(imgc[index1, index2])

axarr[0].imshow(img, cmap="gray")
axarr[0].set_title('Original')

axarr[1].imshow(imgb, cmap="gray")
axarr[1].set_title('img + 100')

axarr[2].imshow(imgc, cmap="gray")
axarr[2].set_title('cv.add')

plt.show()
