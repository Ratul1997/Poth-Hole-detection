from PIL import Image
import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt

image = cv2.imread('poths.png')
road_image = np.copy(image)

hsv_image = cv2.cvtColor(road_image, cv2.COLOR_RGB2HSV)

lower_black1 = np.array([70,10,80])
upper_gray1 = np.array([110,40,220])

lower_black2 = np.array([0,0,0])
upper_gray2 = np.array([130,50,120])

mask1 = cv2.inRange(hsv_image, lower_black1, upper_gray1)
mask2 = cv2.inRange(hsv_image, lower_black2, upper_gray2)
mask = mask1+mask2
res = cv2.bitwise_and(road_image, road_image, mask = mask)

def kmeans(histogram):
    for k in range(0, 21):
        # print '\niteration',k
        ''' First iteration assign random centroid points '''
        if k == 0:
            cent1 = rand_points[0]
            cent2 = rand_points[1]
        else:
            # print '\n selecting centroid values'
            cent1 = centroid1_avg
            cent2 = centroid2_avg

        # print histogram
        point1_centroid = []
        point2_centroid = []
        w1_centroid = []
        w2_centroid = []
        sum1 = 0
        sum2 = 0
        for i, val in enumerate(histogram):
            ''' computing absolute distance from each of the cluster and assigning it to a particular cluster based on distance'''
            # print '\n\n','i',i,'val',val,'cent1', cent1,'cent2', cent2
            if abs(i - cent1) < abs(i - cent2):
                point1_centroid.append(i)
                w1_centroid.append(val)
                sum1 = sum1 + (i * val)

                # print '\nselection 1'
            else:
                point2_centroid.append(i)
                w2_centroid.append(val)
                sum2 = sum2 + (i * val)

                # print '\nselection 2'

        sum_w1 = sum(w1_centroid)
        sum_w2 = sum(w2_centroid)
        if (sum_w1 == 0):
            sum_w1 = 1
        elif (sum_w2 == 0):
            sum_w2 = 1

        centroid1_avg = int(sum1) / sum_w1
        centroid2_avg = int(sum2) / sum_w2

        # print '\n\n','sum1',sum1,'sum2',sum2,'cent1', centroid1_avg,'cent2', centroid2_avg
    return [point1_centroid, point2_centroid]


img1 = np.copy(res)
img = cv2.resize(img1, (256, 256))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

arr = np.asarray(gray)

# k-means clustering

rows, columns = np.shape(arr)

rand_points = [random.randint(1, 255) for i in range(2)]

'''finding the histogram of the image to obtain total number of pixels in each level'''

hist, bins = np.histogram(arr, 256, [1, 256])

# print hist,bins

centroid1_avg = 0
centroid2_avg = 0

res = kmeans(hist)
# print res


end = np.zeros((rows, columns))

if (len(res[1]) < len(res[0])):
    for i in range(rows):
        for j in range(columns):

            if (arr[i][j] in res[1]):
                end[i][j] = int(255)

            else:
                end[i][j] = int(0)

elif (len(res[0]) < len(res[1])):
    for i in range(rows):
        for j in range(columns):

            if (arr[i][j] in res[0]):
                end[i][j] = int(255)

            else:
                end[i][j] = int(0)
# print end

img = np.uint8(end)

# cv2.namedWindow('Segmented', cv2.WINDOW_NORMAL)
# cv2.imshow("Segmented", img)

median = cv2.medianBlur(img,7)
cv2.namedWindow('Median_segmented', cv2.WINDOW_NORMAL)
cv2.imshow("Median_segmented", median)


cv2.waitKey(0)
cv2.destroyAllWindows

#cv2.imwrite('med.JPG', median)
