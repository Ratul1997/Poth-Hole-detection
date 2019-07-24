import numpy as np
import argparse
import cv2
import time
import os
import pickle
import sys
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play

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
        [(30, height), (132, 265), (280, 265),(width, 450),(width,height) ]])

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

    low_red = np.array([0, 0, 0])
    high_red = np.array([179, 43, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)
    cv2.imshow("ss",red)
    return red


def histrigramEqu(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img)


    blur = cv2.GaussianBlur(equ, (5, 5), 0)

    cv2.imshow("ss", equ)
    cv2.imshow("bl", blur)

    TH = cv2.getTrackbarPos("TH", "Trackbar")

    _, res2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("th", res2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosion = cv2.erode(res2, kernel, iterations=2)

    res2 = cv2.dilate(erosion, kernel, iterations=4)

    cv2.imshow("th2", res2)
    res2 = cv2.morphologyEx(res2, cv2.MORPH_OPEN, kernel)

    cv2.imshow("th3", res2)

    return res2


def floodFill(im_th):
    im_floodfill = im_th.copy()

    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0, 0), 0)
    # cv2.floodFill(im_floodfill, mask, (511, 470), 0)

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
        # print(cv2.contourArea(cnt))
        if 500 < cv2.contourArea(cnt) < 3800:
            print("sss " + str(cv2.contourArea(cnt)))
            # cv2.drawContours(frame, [cnt], 0, (0, 255, 0), thickness=cv2.FILLED)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # playsound('Sund/1.mp3')
            # song = AudioSegment.from_mp3('Sund/1.mp3')
            # play(song)
            # cv2.imwrite("output/"+str(cv2.contourArea(cnt))+".jpg",frame)



    cv2.imshow("th4", frame)
    return mask

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

        frame = floodFill(frame)
        frame = contrs(frame,cpy)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()