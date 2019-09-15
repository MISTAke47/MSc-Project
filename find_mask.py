'''
This file is used to find the appropriate mask centre and radius
Assign different centre and radius to see whether we get the appropriate area
Put this file in the main directory, and assign the timestamp manually

Author: Yan Gao
email: gaoy4477@gmail.com
'''

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import module.content as content

# these two lines to solve the bug reported on Mojave 10.14
import matplotlib
matplotlib.use('MacOSX')

# Set different centre to get different mask area
centre = (700, 810)
# Here, centre match the definition in image, but not match the definition in matrix
# In matrix index, we use [row, column], which equals to (column, row) in image
radius = 550
# assign the target timestamp manually
# will automatically pick the first slice
timestamp = '0025'


current_path = os.getcwd()
all_timestamp = content.get_folder(current_path, 'SHP')
timestamp_index = [all_timestamp.index(i) for i in all_timestamp if timestamp in i]
target_timestamp = all_timestamp[timestamp_index[0]]
sub_path = os.path.join(current_path, target_timestamp)
sub_all_tif = content.get_allslice(sub_path)
# choose the first image to find the centre and radius
img_path = sub_all_tif[0]
print('Image:', img_path)

# pick one image, and let user click the two centre to help finding the centre and radius
img = cv2.imread(img_path, -1)
plt.imshow(img, 'gray')
plt.title('Click the centre and the edge successively \n to get recommendation value for centre and radius.')
coordinate = plt.ginput(n=2, timeout=0)
labeled_centre = (int(coordinate[0][0]), int(coordinate[0][1]))
labeled_radius = int(np.sqrt((coordinate[1][0]-coordinate[0][0])**2 + (coordinate[1][1]-coordinate[0][1])**2))
print('Centre:', labeled_centre)
print('Radius:', labeled_radius)
plt.close()

# Here will show two different image, first one is under previous centre and radius, and second one is under labeled centre and radius
# just for easier choosing centre and radius
height, width= img.shape
circle_img = np.zeros((height, width), np.uint8)
circle_img_labeled = np.zeros((height, width), np.uint8)

cv2.circle(circle_img, centre, radius, 1, thickness=-1)
cv2.circle(circle_img_labeled, labeled_centre, labeled_radius, 1, thickness=-1)

masked_data = cv2.bitwise_and(img, img, mask=circle_img)
masked_data_labeled = cv2.bitwise_and(img, img, mask=circle_img_labeled)
plt.figure()
ax = plt.subplot(121)
ax.set_title('previous centre and radius')
plt.imshow(masked_data, 'gray')
ax = plt.subplot(122)
ax.set_title('labeled centre and radius')
plt.imshow(masked_data_labeled, 'gray')
plt.show()





