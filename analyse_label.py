'''
This file is written for find the corresponding label for unsupervised algorithms
It will load one slice and models, then apply the model and return the pictures for 
each class 

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
                        help='File name of saved model for 4D data', default='none')
	parser.add_argument('--model_3D', nargs="?", type=str, 
                        help='File name of saved model for 3D data', default='none')
	parser.add_argument('--size', nargs="?", type=int,
						help='Size of features, should be 1, 3 or 5', default=3)
	parser.add_argument('--timestamp', nargs="?", type=str,
						help='Target timestamp')
	parser.add_argument('--slice', nargs="?", type=int,
						help='Target slice')
	args = parser.parse_args()
	print(args)
	return args

# centre and radius for the mask
mask_centre = (700, 810)
radius = 550
# keyword for choosing the right data
keyword = 'SHP'


# get argument from command line
args = get_args()
# get the path for target slice
current_path = os.getcwd()
all_timestamp = content.get_folder(current_path, keyword)
timestamp_index = [all_timestamp.index(i) for i in all_timestamp if args.timestamp in i]
sub_path = os.path.join(current_path, all_timestamp[timestamp_index[0]])
sub_all_tif = content.get_allslice(sub_path)
target_slice = sub_all_tif[args.slice-1]


# prepare feature vector
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

# load and apply the model from 'model' folder
# if not assign the model, will not apply that model
if args.model_4D != 'none':
	model_4D_path = os.path.join(current_path, 'model', args.model_4D+'.model')
	model_4D_type = load(model_4D_path)
	prediction_4D = model_4D_type.predict(feature_4D)
	num_classes_4D = len(set(prediction_4D))
else:
	num_classes_4D = 0

if args.model_3D != 'none':
	model_3D_path = os.path.join(current_path, 'model', args.model_3D+'.model')
	model_3D_type = load(model_3D_path)
	prediction_3D = model_3D_type.predict(feature_3D)
	num_classes_3D = len(set(prediction_3D))
else:
	num_classes_3D = 0


# write the image
# obtain the coordinate and the image size from mask
coordinate = mask.nonzero()
height, width = mask.shape

# plot each class for 4D and 3D model
if num_classes_4D != 0:
	for i in range(num_classes_4D):
		final_img_4D = np.ones((height,width), np.uint8)
		zero_point_4D_co = np.argwhere(prediction_4D==i)
		for j in zero_point_4D_co:
			final_img_4D[coordinate[0][j], coordinate[1][j]] = 0

		# plot the picture
		plt.figure()
		plt.imshow(final_img_4D, 'gray')
		plt.axis('off')
		plt.title('Segment for 4D data, class {:d}'.format(i))
		# comment for ploting
		# name_4D = 'analyse_label_4D_'+str(i)+'.png'
		# plt.savefig(name_4D, bbox_inches='tight', pad_inches=0.0)

if num_classes_3D != 0:
	for i in range(num_classes_3D):
		final_img_3D = np.ones((height,width), np.uint8)
		zero_point_3D_co = np.argwhere(prediction_3D==i)
		for j in zero_point_3D_co:
			final_img_3D[coordinate[0][j], coordinate[1][j]] = 0

		# plot the picture
		plt.figure()
		plt.imshow(final_img_3D, 'gray')
		plt.axis('off')
		plt.title('Segment for 3D data, class {:d}'.format(i))
		# name_3D = 'analyse_label_3D_'+str(i)+'.png'
		# plt.savefig(name_3D, bbox_inches='tight', pad_inches=0.0)

# plot the original slice
plt.figure()
img = cv2.imread(target_slice, -1)
plt.imshow(img, 'gray')
plt.axis('off')
plt.title('Original slice')
# plt.savefig('original_image.png', bbox_inches='tight', pad_inches=0.0)

plt.show()
