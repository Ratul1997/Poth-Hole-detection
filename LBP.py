import cv2
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
from skimage import feature

def describe(image,numPoints,radius, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, numPoints,
            radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, numPoints + 3),
            range=(0, numPoints + 2))
 
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 
        # return the histogram of Local Binary Patterns
        return hist,lbp
if __name__ == '__main__':
    
    
    while True:
        frame = cv2.imread("poths.png")


        frame = cv2.resize(frame,(512,512))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist,lbp = describe(gray,10,5)
        
        cv2.imshow("s",frame)
        print(lbp)

        cv2.imshow("his",lbp.astype("uint8"))

        k = cv2.waitKey(30)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        
cv2.destroyAllWindows()
        
    
