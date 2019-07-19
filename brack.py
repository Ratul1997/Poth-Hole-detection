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



def describe(image,numPoints,radius, eps=1e-7):
        lbp = feature.local_binary_pattern(image, numPoints,
            radius, method="default")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, numPoints + 3),
            range=(0, numPoints + 2))
 
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 	
 		# cv2.imshow("lbp",lbp.astype("uint8"))
        return hist,lbp

def cannys(gray):

	gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)
	canny = cv2.Canny(blur,100,150)
	return canny,blur

def histrigramEqu(img):

	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	print(np.array(img))
	equ = cv2.equalizeHist(img)
	# res = np.hstack((img,equ))

	blur = cv2.GaussianBlur(equ,(5,5),0)

	cv2.imshow("ss",equ)
	cv2.imshow("bl",blur)

	TH = cv2.getTrackbarPos("TH","Trackbar")

	_, res2 = cv2.threshold(blur, TH, 255, cv2.THRESH_BINARY_INV)


	cv2.imshow("th",res2)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	erosion = cv2.erode(res2,kernel,iterations = 2)

	res2 = cv2.dilate(erosion,kernel,iterations = 4)
	
	cv2.imshow("th2",res2)
	res2 = cv2.morphologyEx(res2, cv2.MORPH_OPEN, kernel)	

	cv2.imshow("th3",res2)

def kmeans(img):
	# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	Z = img.reshape((-1,3))

	# convert to np.float32
	Z = np.float32(Z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 8
	ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))

	cv2.imshow("km",res2)

def morphology(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img,(5,5),0)

	TH = cv2.getTrackbarPos("TH","Trackbar")

	_, img = cv2.threshold(img, TH, 255, cv2.THRESH_BINARY)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

	erosion = cv2.erode(img,kernel,iterations = 2)

	dilation = cv2.dilate(erosion,kernel,iterations = 4)

	opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

	closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)

	gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

	tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

	blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

	# cv2.imshow("erosion",erosion)

	cv2.imshow("dilate",dilation)
	cv2.imshow("opening",opening)
	cv2.imshow("closing",closing)
	cv2.imshow("gradient",gradient)
	cv2.imshow("tophat",tophat)
	cv2.imshow("blackhat",blackhat)

def cropImage(frame):
    height, width, _ = frame.shape
    points = np.array([
        [(0, height), (127, 220), (265, 220),(width, 400),(width,height) ]])

    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, np.int32([points]), True, 255, 2)
    cv2.fillPoly(mask, np.int32([points]), 255)
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_image

if __name__ == '__main__':

	while True:
		frame = cv2.imread('158.jpg')

		frame = cv2.resize(frame, (512, 512))

		hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		canny, blur = cannys(frame)
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

		# morphology(frame)

		img = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
		# kmeans(red)
		histrigramEqu(red)
		# cv2.imshow("blur",blur)
		# cv2.imshow("canny",canny)
		hist, lbp = describe(img, 24, 8)
		#
		cv2.imshow("lbp", lbp.astype("uint8"))
		cv2.imshow("frame", red)

		k = cv2.waitKey(0)
		if k % 256 == 27:
			print("Escape hit, closing...")
			break

	cv2.destroyAllWindows()

    
