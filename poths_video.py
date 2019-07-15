import skimage
import numpy as np
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
from random import shuffle
import cv2
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


 
cap = cv2.VideoCapture('v2.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    frame = cv2.resize(frame,(512,512))
    retval, threshold = cv2.threshold(frame,200,255,cv2.THRESH_BINARY)

    grayScaled = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    retval2, threshold2 = cv2.threshold(grayScaled,125,255,cv2.THRESH_BINARY_INV)
    gauss = cv2.adaptiveThreshold(grayScaled,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)

  	
    # hsv_frame = cv2.cvtColor(cv2.resize(frame,(280,280)), cv2.COLOR_BGR2HSV)

    # l_h = cv2.getTrackbarPos("L-H","Trackbar")
    # l_s = cv2.getTrackbarPos("L-S","Trackbar")
    # l_v = cv2.getTrackbarPos("L-V","Trackbar")
    # u_h = cv2.getTrackbarPos("U-H","Trackbar")
    # u_s = cv2.getTrackbarPos("U-S","Trackbar")
    # u_v = cv2.getTrackbarPos("U-V","Trackbar")

    # low_red = np.array([l_h, l_s, l_v])
    # high_red = np.array([u_h, u_s, u_v])
    # red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    # red = cv2.bitwise_and(frame, frame, mask=red_mask)

    # # # cv2.imshow("Frame", gray)
    canny = cv2.Canny(threshold2,200,200)
    
    cv2.imshow('Frame',frame)
    # cv2.imshow('threshold',threshold)
    cv2.imshow('threshold2',threshold2)
    # cv2.imshow("gauss",gauss)
    cv2.imshow("canny",canny)
    
    # cv2.imshow("",red_mask)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()