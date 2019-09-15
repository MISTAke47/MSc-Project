'''
This file will select one time stamp, one slice to show the segmentation result

Author: Yan Gao
email: gaoy4477@gmail.com
'''

import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt 
import module.content as content
import module.features as features
from joblib import load
import argparse
import time

# these two lines to solve the bug reported on Mojave 10.14
import matplotlib
matplotlib.use('MacOSX')

def get_args():
	parser = argparse.ArgumentParser(description='Show single results')

	parser.add_argument('--model_4D', nargs="?", type=str, 
                        help='File name of saved model for 4D data')
	parser.add_argument('--model_3D', nargs="?", type=str, 
                        help='File name of saved model for 3D data')
	parser.add_argument('--size', nargs="?", type=int,
						help='Size of features, should be 1, 3 or 5')
	parser.add_argument('--timestamp', nargs="?", type=str,
						help='Target timestamp')
	parser.add_argument('--slice', nargs="?", type=int,
						help='Target slice')
	parser.add_argument('--pore_4D', nargs="?", type=str,
						help='Label for pore in 4D model')
	parser.add_argument('--pore_3D', nargs="?", type=str,
						help='Label for pore in 3D model')
	args = parser.parse_args()
	print(args)
	return args


args = get_args()

# Here we set the paramater
mask_centre = (700, 810)
radius = 550
keyword = 'SHP'
# transfer the pore from string to list
pore_4D = args.pore_4D.split(',')
pore_4D = [int(i) for i in pore_4D]
pore_3D = args.pore_3D.split(',')
pore_3D = [int(i) for i in pore_3D]

# get the path for target slice
current_path = os.getcwd()
all_timestamp = content.get_folder(current_path, keyword)
timestamp_index = [all_timestamp.index(i) for i in all_timestamp if args.timestamp in i]
sub_path = os.path.join(current_path, all_timestamp[timestamp_index[0]])
sub_all_tif = content.get_allslice(sub_path)
target_slice = sub_all_tif[args.slice-1]

# load the model from 'model' folder
model_4D_path = os.path.join(current_path, 'model', args.model_4D+'.model')
model_3D_path = os.path.join(current_path, 'model', args.model_3D+'.model')
model_4D_type = load(model_4D_path)
model_3D_type = load(model_3D_path)

# get features
mask, feature_index = features.get_mask(sub_all_tif[0], mask_centre, radius, args.size)
if args.size == 1:
	feature_4D, feature_3D = features.get_all_features_1(target_slice, feature_index, keyword)
elif args.size == 3:
	feature_4D, feature_3D = features.get_all_features_3(target_slice, feature_index, keyword)
elif args.size == 5:
	feature_4D, feature_3D = features.get_all_features_5(target_slice, feature_index, keyword)
else:
	raise ValueError('Please input the right size, should be 1, 3 or 5.')

print('Segmenting...')
# segment
prediction_4D = model_4D_type.predict(feature_4D)
prediction_3D = model_3D_type.predict(feature_3D)

# write the image

coordinate = mask.nonzero()

height, width = mask.shape
final_img_4D = np.ones((height,width), np.uint8)
final_img_3D = np.ones((height,width), np.uint8)

# here need to assign the pore label manually. 
# Since the classfier will return the label randomly
for element in pore_4D:
	zero_point_4D_co = np.argwhere(prediction_4D==element)
	for i in zero_point_4D_co:
		final_img_4D[coordinate[0][i], coordinate[1][i]] = 0

for element in pore_3D:
	zero_point_3D_co = np.argwhere(prediction_3D==element)
	for j in zero_point_3D_co:
		final_img_3D[coordinate[0][j], coordinate[1][j]] = 0

# plot the picture
plt.figure()
plt.imshow(final_img_4D, 'gray')
plt.axis('off')
plt.title('Segment for 4D data')

plt.figure()
plt.imshow(final_img_3D, 'gray')
plt.axis('off')
plt.title('Segment for 3D data')

plt.figure()
img = cv2.imread(target_slice, -1)
plt.imshow(img, 'gray')
plt.title('Original slice')

plt.show()






