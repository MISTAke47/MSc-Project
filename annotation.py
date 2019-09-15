'''
This file return the list for all annotated points, named as 'filename.txt', and save it in './validation_data' folder
We need to give the 'filename' for the text file

This text file save the information as format below:
Path -- coordinate -- label

Author: Yan Gao
email: gaoy4477@gmail.com
'''
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import module.content as content

# these two lines to solve the bug reported on Mojave 10.14
import matplotlib
matplotlib.use('MacOSX')

# setting the parameter here
# choose specific time stamp
begin_timestamp= '0025'
end_timestamp = '0025'
# keyword for target folder
keyword = 'SHP'
# set the range for randomly choosing slice
begin_slice = 600
end_slice = 800
# set the number of slices
num_slices = 50
# set the filename
filename = 'labeled_data_0025_3classes'
# area for show the lable image
# set 100 will show the 200x200 area
show_length = 100
# set the mask centre and radius for each slice
# we need to know it before annotation, use find_mask.py to determine the centre and radius
mask_centre = (700, 810)
radius = 550
# set the number of points for each slice, both for pore and non-pore
# -1 means point any points you want, until you press 'Enter'
num_points = -1

# add mask for original image and return the masked image
def add_mask(centre, radius, source_image):
	height, width = source_image.shape
	circle_img = np.zeros((height,width), np.uint8)
	cv2.circle(circle_img,centre,radius,1,thickness=-1)
	masked_data = cv2.bitwise_and(source_image, source_image, mask=circle_img)
	return masked_data

# random select one point from the valid area
def random_effective_area(masked_image):
	height, width = masked_image.shape
	flag = 1
	while(flag):
		x_co = random.randint(0, height-1)
		y_co = random.randint(0, width-1)
		if masked_image[x_co, y_co] != 0:
			flag = 0
	return x_co, y_co

# transform the coordinate 
def transform(coordinate, x_coordinate, y_coordinate, length):
	transformed_coordinate = [(element[1]+x_coordinate-length, element[0]+y_coordinate-length) for element in coordinate]
	return transformed_coordinate


# main program
print('Prepare annotation')
current_path = os.getcwd()
# get all data folder
all_timestamp = content.get_folder(current_path, keyword)
# get the index for begin and end
begin_timestamp_index = [all_timestamp.index(i) for i in all_timestamp if begin_timestamp in i]
end_timestamp_index = [all_timestamp.index(i) for i in all_timestamp if end_timestamp in i]
# create the target data
sample_timestamp = all_timestamp[begin_timestamp_index[0]:(end_timestamp_index[0]+1)]

# check if it exists the directory to save data
# data will save in ./validation_data as filename.txt
save_folder = os.path.join(current_path, 'validation_data')
if not os.path.exists(save_folder):
	os.mkdir(save_folder)
file_path = os.path.join(save_folder, filename+'.txt')

# before labeling the data, delete the file with same name
if os.path.exists(file_path):
	os.remove(file_path)
print('Finished!')

# begin annotation
print('Will choose timestamp from {begin} to {end}'.format(begin=all_timestamp[begin_timestamp_index[0]], end=all_timestamp[end_timestamp_index[0]]))
print('Will randomly choose {:d} slices from slice {:d} to {:d}'.format(num_slices, begin_slice, end_slice))
print('Left click to add points, and right click to remove the mose recently added points. Press enter to label next slice.')

for sub_timestamp in sample_timestamp:
	sub_path = os.path.join(current_path, sub_timestamp)
	sub_all_tif = content.get_allslice(sub_path)

	# First, we label all pore
	print('Please label any points for pore in each picture!')
	random_slice_pore = np.random.randint(begin_slice-1, end_slice, num_slices)
	for index, i in enumerate(random_slice_pore):
		slice_path = sub_all_tif[i-1]
		slice_img = cv2.imread(slice_path, -1)
		# now get the slice image
		masked_image = add_mask(mask_centre, radius, slice_img)
		x_coordinate, y_coordinate = random_effective_area(masked_image)

		plt.imshow(masked_image[x_coordinate-show_length:x_coordinate+show_length, y_coordinate-show_length:y_coordinate+show_length], 'gray')
		plt.title('Please label any points for pore! ({:d}/{:d}) \n Current slice: {str}'.format((index+1), num_slices, str=os.path.basename(slice_path)), color='red')

		coordinate = plt.ginput(n=num_points, timeout=0)

		# note that x, y from the ginput function is oppisite to our x and y, we need to transfer it
		transformed_coordinate = transform(coordinate, x_coordinate, y_coordinate, show_length)
		print(transformed_coordinate)
		
		with open(file_path, 'a') as f:
			f.writelines([slice_path, ' ', str(transformed_coordinate), ' ', '0', '\n'])
			# '0' means pore

	print('Please label any points for non-pore (not artifact) in each picture!')
	# Then we label all non-pore
	random_slice_nonpore = np.random.randint(begin_slice-1, end_slice, num_slices)
	for index, i in enumerate(random_slice_nonpore):
		slice_path = sub_all_tif[i-1]
		slice_img = cv2.imread(slice_path, -1)
		masked_image = add_mask(mask_centre, radius, slice_img)
		x_coordinate, y_coordinate = random_effective_area(masked_image)

		plt.imshow(masked_image[x_coordinate-show_length:x_coordinate+show_length, y_coordinate-show_length:y_coordinate+show_length], 'gray')
		plt.title('Please label any points for non-pore (non-artifact)! ({:d}/{:d}) \n Current slice: {str}'.format((index+1), num_slices, str=os.path.basename(slice_path)), color='red')

		coordinate = plt.ginput(n=num_points, timeout=0)

		transformed_coordinate = transform(coordinate, x_coordinate, y_coordinate, show_length)
		print(transformed_coordinate)

		with open(file_path,'a') as f:
			f.writelines([slice_path, ' ', str(transformed_coordinate), ' ', '1', '\n'])
			# '1' means non-pore (not artifact)

	print('Please label any points for non-pore (artifact) in each picture!')
	# Then we label all non-pore
	random_slice_nonpore = np.random.randint(begin_slice-1, end_slice, num_slices)
	for index, i in enumerate(random_slice_nonpore):
		slice_path = sub_all_tif[i-1]
		slice_img = cv2.imread(slice_path, -1)
		masked_image = add_mask(mask_centre, radius, slice_img)
		# assign the centre for specific area
		x_coordinate = 820
		y_coordinate = 820

		plt.imshow(masked_image[x_coordinate-show_length:x_coordinate+show_length, y_coordinate-show_length:y_coordinate+show_length], 'gray')
		plt.title('Please label any points for non-pore (artifact)! ({:d}/{:d}) \n Current slice: {str}'.format((index+1), num_slices, str=os.path.basename(slice_path)), color='red')

		coordinate = plt.ginput(n=num_points, timeout=0)

		transformed_coordinate = transform(coordinate, x_coordinate, y_coordinate, show_length)
		print(transformed_coordinate)

		with open(file_path,'a') as f:
			f.writelines([slice_path, ' ', str(transformed_coordinate), ' ', '2', '\n'])
			# '2' means non-pore (artifact)
			
print('Finished!')



