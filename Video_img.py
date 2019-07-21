import numpy as np
import argparse
import cv2
import time
import os
import pickle

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
        [(0, height), (135, 260), (290, 260), (width, 450), (width, height)]])

    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, np.int32([points]), True, 255, 2)
    cv2.fillPoly(mask, np.int32([points]), 255)
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_image


def histrigramEqu(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    equ = cv2.equalizeHist(img)
    # res = np.hstack((img,equ))

    blur = cv2.GaussianBlur(equ, (5, 5), 0)

    cv2.imshow("ss", equ)
    cv2.imshow("bl", blur)

    TH = cv2.getTrackbarPos("TH", "Trackbar")

    _, res2 = cv2.threshold(blur, TH, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("th", res2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosion = cv2.erode(res2, kernel, iterations=2)

    res2 = cv2.dilate(erosion, kernel, iterations=4)

    cv2.imshow("th2", res2)
    res2 = cv2.morphologyEx(res2, cv2.MORPH_OPEN, kernel)

    cv2.imshow("th3", res2)
    print(np.array(res2))


def cnvrtHSV(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H", "Trackbar")
    l_s = cv2.getTrackbarPos("L-S", "Trackbar")
    l_v = cv2.getTrackbarPos("L-V", "Trackbar")
    u_h = cv2.getTrackbarPos("U-H", "Trackbar")
    u_s = cv2.getTrackbarPos("U-S", "Trackbar")
    u_v = cv2.getTrackbarPos("U-V", "Trackbar")

    low_red = np.array([0, 0, 0])
    high_red = np.array([179, 43, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)
    cv2.imshow("ss",red)
    return red


def histrigramEqu(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(np.array(img))
    equ = cv2.equalizeHist(img)
    # res = np.hstack((img,equ))

    blur = cv2.GaussianBlur(equ, (5, 5), 0)

    cv2.imshow("ss", equ)
    cv2.imshow("bl", blur)

    TH = cv2.getTrackbarPos("TH", "Trackbar")

    _, res2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)

    cv2.imshow("th", res2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosion = cv2.erode(res2, kernel, iterations=2)

    res2 = cv2.dilate(erosion, kernel, iterations=4)

    cv2.imshow("th2", res2)
    res2 = cv2.morphologyEx(res2, cv2.MORPH_OPEN, kernel)

    cv2.imshow("th3", res2)

    return res2


def contrs(img,cpy):
    mask = np.zeros(img.shape, np.uint8)
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        print(cv2.contourArea(cnt))
        if 200 < cv2.contourArea(cnt) < 110500:
            print("sss " + str(cv2.contourArea(cnt)))
            cv2.drawContours(cpy, [cnt], 0, (0, 255, 0), 2)
            cv2.drawContours(mask, [cnt], 0, 255, -1)

    cv2.imshow("th4", mask)
    cv2.bitwise_not(img, img, mask)
    cv2.imshow('IMG', cpy)
    return img

cap = cv2.VideoCapture('video/1.mp4')

if (cap.isOpened() == False):
    print("Error opening video stream or file")

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame,(512,512))

        cpy = frame

        cv2.imshow("frames",frame)
        frame = cropImage(frame)

        frame = cnvrtHSV(frame)

        cv2.imshow('Frame', frame)

        frame = histrigramEqu(frame)
        frame = contrs(frame,cpy)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()