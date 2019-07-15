from skimage import feature

import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2

def resizeImage(image):
    
    h, w, _ = image.shape

    width = 360  #  This "width" is the width of the resize`ed image
    # calculate the ratio of the width and construct the
    # dimensions
    ratio = width / float(w)
    dim = (width, int(h * ratio))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    #resized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
    return resized

# 1 load the image
imagepath = "Pothole.jpg"
# , double it in size, and grab the cell size
image = cv2.imread(imagepath)


image = resizeImage(image)
h, w,_ = image.shape
cellSize = h/10

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image", gray)
cv2.waitKey(0)

cv2.imwrite("docsIMG/gray_resized_image.png", gray)

# construct the figure
plt.style.use("ggplot")
(fig, ax) = plt.subplots()
fig.suptitle("Local Binary Patterns")
plt.ylabel("% of Pixels")
plt.xlabel("LBP pixel bucket")


# displaying default to make cool image
features = feature.local_binary_pattern(gray, 30, 10, method="default") # method="uniform")
cv2.imshow("LBP", features.astype("uint8"))
cv2.waitKey(0)


ax.hist(features.ravel(), normed=True, bins=20, range=(0, 256))
ax.set_xlim([0, 256])
ax.set_ylim([0, 0.030])
# save figure
fig.savefig('docsIMG/lbp_histogram.png')   # save the figure to file
plt.show()

cv2.destroyAllWindows()
#######################################################################

stacked = np.dstack([gray]* 3)

# Divide the image into 100 pieces
(h, w) = stacked.shape[:2]
cellSizeYdir = h / 10
cellSizeXdir = w / 10

# Draw the box around area
# loop over the x-axis of the image
for x in range(0, int(w), int(cellSizeXdir)):
    # draw a line from the current x-coordinate to the bottom of
    # the image

    cv2.line(stacked, (x, 0), (x, h), (0, 255, 0), 1)
    #   
# loop over the y-axis of the image
for y in range(0, int(h), int(cellSizeYdir)):
    # draw a line from the current y-coordinate to the right of
    # the image
    cv2.line(stacked, (0, y), (w, y), (0, 255, 0), 1)

# draw a line at the bottom and far-right of the image
cv2.line(stacked, (0, h - 1), (w, h - 1), (0, 255, 0), 1)
cv2.line(stacked, (w - 1, 0), (w - 1, h - 1), (0, 255, 0), 1)

