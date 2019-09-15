'''
This file will select one time stamp, and apply segmentation algorithm to all the slices

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

# this function get args for segmentation
def get_args():
	parser = argparse.ArgumentParser(description='Show results')

	parser.add_argument('--model_4D', nargs="?", type=str, default='none',
                        help='File name of saved model for 4D data')
	parser.add_argument('--model_3D', nargs="?", type=str, default='none',
                        help='File name of saved model for 3D data')
	parser.add_argument('--size', nargs="?", type=int, default=3,
						help='Size of features, should be 1, 3 or 5')
	parser.add_argument('--timestamp', nargs="?", type=str,
						help='Target timestamp')
	parser.add_argument('--pore_4D', nargs="?", type=str,
						help='Label for pore in 4D model')
	parser.add_argument('--pore_3D', nargs="?", type=str,
						help='Label for pore in 3D model')
	args = parser.parse_args()
	print(args)
	return args

# function for saving the .png file
def save_png(raw_img_path, save_folder, img_data):
	# plt.figure(figsize=(height/1000, width/1000), dpi=100)
	# plt.imshow(img_data, 'gray')
	# plt.axis('off')
	img_data = img_data * 255
	save_path = os.path.join(save_folder, os.path.basename(raw_img_path[:-9])+'8bit.png')
	# plt.savefig(save_path, dpi=1000)
	# plt.close()
	cv2.imwrite(save_path, img_data)


def segment(path_img, save_path_4D, save_path_3D, model_4D, model_3D,
			mask, feature_index, size, pore_4D, pore_3D, keyword, flag_4D, flag_3D):
	start = time.time()
	# record the time
	if size == 1:
		feature_4D, feature_3D = features.get_all_features_1(path_img, feature_index, keyword)
	elif size == 3:
		feature_4D, feature_3D = features.get_all_features_3(path_img, feature_index, keyword)
	elif size == 5:
		feature_4D, feature_3D = features.get_all_features_5(path_img, feature_index, keyword)
	else:
		raise ValueError('Please input the right size, should be 1, 3 or 5.')

	coordinate = mask.nonzero()
	height, width = mask.shape
	
	
	if flag_4D:
		print('4D Segmenting and ploting...')
		prediction_4D = model_4D.predict(feature_4D)
		final_img_4D = np.ones((height,width), np.uint8)

		for element in pore_4D:
			zero_point_4D_co = np.argwhere(prediction_4D==element)
			for i in zero_point_4D_co:
				final_img_4D[coordinate[0][i], coordinate[1][i]] = 0

		# save image
		save_png(path_img, save_path_4D, final_img_4D)

		print('Finished')

	if flag_3D:
		print('3D Segmenting and ploting...')
		prediction_3D = model_3D.predict(feature_3D)
		final_img_3D = np.ones((height,width), np.uint8)

		for element in pore_3D:
			zero_point_3D_co = np.argwhere(prediction_3D==element)
			for j in zero_point_3D_co:
				final_img_3D[coordinate[0][j], coordinate[1][j]] = 0
		# save image
		save_png(path_img, save_path_3D, final_img_3D)

		print('Finished!')	

	end = time.time()
	print(end-start)


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

current_path = os.getcwd()
all_timestamp = content.get_folder(current_path, keyword)
timestamp_index = [all_timestamp.index(i) for i in all_timestamp if args.timestamp in i]
sub_path = os.path.join(current_path, all_timestamp[timestamp_index[0]])
sub_all_tif = content.get_allslice(sub_path)

# assign the target document
document_path_4D = os.path.join(os.path.dirname(sub_all_tif[0]),'segmentation_4D')
if not os.path.exists(document_path_4D):
	os.mkdir(document_path_4D)
document_path_3D = os.path.join(os.path.dirname(sub_all_tif[0]),'segmentation_3D')
if not os.path.exists(document_path_3D):
	os.mkdir(document_path_3D)



# just pick one slice to get the mask and its corresponding features index
mask, feature_index = features.get_mask(sub_all_tif[0], mask_centre, radius, args.size)



print('Will segment', len(sub_all_tif), 'slices')
if flag_model_4D == 0:
	if flag_model_3D == 0:
		raise ValueError('No model for segmentation!')
for i in sub_all_tif:
	segment(i, document_path_4D, document_path_3D, model_4D_type, model_3D_type, 
			mask, feature_index, args.size, pore_4D, pore_3D, keyword, flag_model_4D, flag_model_3D)












