import cv2
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt1
from skimage import feature
import matplotlib.pyplot as plt2

def nothing(x):
	pass


cv2.namedWindow("Trackbar")

cv2.createTrackbar("L-H", "Trackbar", 0 , 179, nothing)
cv2.createTrackbar("L-s", "Trackbar", 0 , 255, nothing)
cv2.createTrackbar("L-V", "Trackbar", 0 , 255, nothing)
cv2.createTrackbar("U-H", "Trackbar", 179 , 179, nothing)
cv2.createTrackbar("U-S", "Trackbar", 255 , 255, nothing)
cv2.createTrackbar("U-V", "Trackbar", 255 , 255, nothing)
cv2.createTrackbar("TH", "Trackbar", 0 , 255, nothing)


def cropImage(frame):
    height, width, _ = frame.shape
    points = np.array([
        [(15, height), (127, 260), (290, 260),(width, 450),(width,height) ]])

    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, np.int32([points]), True, 255, 2)
    cv2.fillPoly(mask, np.int32([points]), 255)
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_image

def cnvrtHSV(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H", "Trackbar")
    l_s = cv2.getTrackbarPos("L-S", "Trackbar")
    l_v = cv2.getTrackbarPos("L-V", "Trackbar")
    u_h = cv2.getTrackbarPos("U-H", "Trackbar")
    u_s = cv2.getTrackbarPos("U-S", "Trackbar")
    u_v = cv2.getTrackbarPos("U-V", "Trackbar")

    blur = cv2.GaussianBlur(hsv_frame, (5, 5), 0)

    edges = cv2.Canny(blur, 100, 200)

    low_red = np.array([l_h, l_s, l_v])
    high_red = np.array([u_h, u_s, u_v])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)
    cv2.imshow("Res",red)
    return red


def histrigramEqu(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(np.array(img))
    equ = cv2.equalizeHist(img)
    # res = np.hstack((img,equ))

    blur = cv2.GaussianBlur(equ, (5, 5), 0)

    # edges = cv2.Canny(blur, 100, 200)

    cv2.imshow("ss", equ)
    # cv2.imshow("bl", edges)

    TH = cv2.getTrackbarPos("TH", "Trackbar")

    _, res2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosion = cv2.erode(res2, kernel, iterations=2)

    res2 = cv2.dilate(erosion, kernel, iterations=4)

    res2 = cv2.morphologyEx(res2, cv2.MORPH_OPEN, kernel)

    cv2.imshow("th3", res2)

    return res2


def floodFill(im_th):
    im_floodfill = im_th.copy()

    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    print(im_th[511][450])
    cv2.floodFill(im_floodfill, mask, (0, 0), 0);
    cv2.floodFill(im_floodfill, mask, (511, 511), 0);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    # Display images.
    cv2.imshow("Thresholded Image", im_th)
    cv2.imshow("Floodfilled Image", im_floodfill)
    return im_floodfill

def contrs(img,frame):
    mask = np.zeros(img.shape, np.uint8)
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        print(cv2.contourArea(cnt))
        if 200 < cv2.contourArea(cnt) < 120380:
            print("sss " + str(cv2.contourArea(cnt)))
            cv2.drawContours(frame, [cnt], 0, (0, 255, 0), thickness=cv2.FILLED)



    cv2.imshow("th4", frame)
    return mask

def Cany(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 100, 150)
    # cv2.imshow("ca",canny)
    floodFill(canny)


while True:
    frame = cv2.imread("output/videos377.jpg")
    cpy_frame = frame
    frame = cv2.resize(frame,(512,512))
    cpy_frame = frame
    cv2.imshow("frame",frame)
    frame = cropImage(frame)

    frame = cnvrtHSV(frame)

    # cv2.imshow("frame2",frame)
    frame = histrigramEqu(frame)
    cv2.imshow("ssas",frame)
    frame = floodFill(frame)
    mask = contrs(frame,cpy_frame)

    k = cv2.waitKey(0)
    if k % 256 == 27:
        print("Escape hit, closing...")
        break


cv2.destroyAllWindows()