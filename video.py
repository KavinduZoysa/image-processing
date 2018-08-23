import cv2 as cv
cap = cv.VideoCapture('videos/conveyor_crpped.m4v')

fgbg = cv.createBackgroundSubtractorMOG2()

while 1:
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv.GaussianBlur(fgmask, (49, 49), 4)
    kernel_size = 5
    fgmask = cv.medianBlur(fgmask, kernel_size)

    im2, contours, hierarchy = cv.findContours(fgmask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    cv.drawContours(frame, contours, -1, (255, 0, 0), 3)
    cv.imshow('frame', frame)

    if cv.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
