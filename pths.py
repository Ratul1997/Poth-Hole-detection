import cv2
import numpy as np 

img = cv2.imread("ppp.png")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures = 4500)
# kp, = sift.detect(img, None)

kp,descriptors = orb.detectAndCompute(img, None)

img = cv2.drawKeypoints(img,kp,None)


cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()