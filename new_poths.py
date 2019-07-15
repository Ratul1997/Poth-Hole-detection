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


def describe(image,numPoints,radius, eps=1e-7):
        
        lbp = feature.local_binary_pattern(image, numPoints,
            radius, method="default")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, numPoints + 3),
            range=(0, numPoints + 2))
 
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 
        return hist,lbp

def cannys(gray):
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)
	canny = cv2.Canny(blur,100,150)
	return canny

def roi(frame):
	height, width, _ = frame.shape
	points = np.array([
		[(0,height),(120,350),(220,350),(470,height)] ])

	
	mask = np.zeros((height, width), np.uint8)
	cv2.polylines(mask, np.int32([points]), True, 255, 2)
	cv2.fillPoly(mask, np.int32([points]), 255)
	masked_image = cv2.bitwise_and(frame, frame, mask=mask)

	

	return masked_image


if __name__ == '__main__':
    
    
    while True:
        frame = cv2.imread("aa.png")


        frame = cv2.resize(frame,(512,512))
        cv2.imshow("frames",frame)
        load_frame = np.copy(roi(frame))


        gray = cv2.cvtColor(load_frame, cv2.COLOR_BGR2GRAY)
        canny = cannys(gray)

        # frame = cv2.GaussianBlur(frame,(5,5),0)
        hsv_frame = cv2.cvtColor(load_frame, cv2.COLOR_BGR2HSV)



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
        red = cv2.bitwise_and(load_frame, load_frame, mask=red_mask)


        gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
        hist,lbp = describe(gray,24,8)

        cv2.imshow("lbp",lbp.astype("uint8"))
        
        # canny2 = cannys(lbp.astype("uint8"))
        cv2.imshow("canny",canny)
        cv2.imshow("s",red)
        img = lbp.astype('uint8')

        canny2 = cannys(img)
        cv2.imshow("canny2",img)

        k = cv2.waitKey(30)

        if k%256 == 27:
            print("Escape hit, closing...")
            break
        
cv2.destroyAllWindows()
        
    
