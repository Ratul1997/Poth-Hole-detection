import cv2
import pickle
import numpy as np
import os

def nothing(x):
	pass


cv2.namedWindow("Trackbar")

cv2.createTrackbar("L-H", "Trackbar", 0 , 179, nothing)
cv2.createTrackbar("L-s", "Trackbar", 0 , 255, nothing)
cv2.createTrackbar("L-V", "Trackbar", 0 , 255, nothing)
cv2.createTrackbar("U-H", "Trackbar", 179 , 179, nothing)
cv2.createTrackbar("U-S", "Trackbar", 255 , 255, nothing)
cv2.createTrackbar("U-V", "Trackbar", 255 , 255, nothing)
        
if __name__ == '__main__':
    
    
    while True:
        frame = cv2.imread('112.png')


        frame = cv2.resize(frame,(512,512))
        retval, threshold = cv2.threshold(frame,100,255,cv2.THRESH_BINARY)

        grayScaled = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        retval2, threshold2 = cv2.threshold(grayScaled,125,255,cv2.THRESH_BINARY)
        gauss = cv2.adaptiveThreshold(threshold2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)

        # frame = cv2.GaussianBlur(frame,(5,5),0)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



        l_h = cv2.getTrackbarPos("L-H","Trackbar")
        l_s = cv2.getTrackbarPos("L-S","Trackbar")
        l_v = cv2.getTrackbarPos("L-V","Trackbar")
        u_h = cv2.getTrackbarPos("U-H","Trackbar")
        u_s = cv2.getTrackbarPos("U-S","Trackbar")
        u_v = cv2.getTrackbarPos("U-V","Trackbar")


        # low_red = np.array([0, 16, 13])
        # high_red = np.array([30, 53, 195])
        low_red = np.array([l_h, l_s, l_v])
        high_red = np.array([u_h, u_s, u_v])
        red_mask = cv2.inRange(hsv_frame, low_red, high_red)
        red = cv2.bitwise_and(frame, frame, mask=red_mask)

        grayScaled = cv2.cvtColor(red,cv2.COLOR_BGR2GRAY)
        retval2, threshold2 = cv2.threshold(grayScaled,125,255,cv2.THRESH_BINARY)
        
       	# canny = cv2.Canny(gauss,250,250)
        # cv2.imshow("canny", canny)
        cv2.imshow("Red", red)
        # cv2.imshow('Frame',frame)
        # cv2.imshow('threshold',threshold)
        cv2.imshow('threshold2',threshold2)
        # cv2.imshow("gauss",gauss)
        k = cv2.waitKey(30)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        
cv2.destroyAllWindows()
        
    
