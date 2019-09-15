'''
This file return the feature for all annotated points
Before running this file, we need to run annotation.py, which will create a text file in './validation_data' folder
This file will return two dataset in './validation_data' folder, for 3D data and 4D data. The format will be 'filename.npy', and we need to give 'filename'
It can return the features with 1x1x1(x3), 3x3x3(x3), 5x5x5(x3) 

Author: Yan Gao
email: gaoy4477@gmail.com
'''
import re
import numpy as np 
import os
import module.features as features

# use this function to save the obtained validation features (features with label)
def save_data(filename, data):
	file_path = os.path.join(os.getcwd(), 'validation_data', filename)
	# before saving the data, delete the old data with the same name
	if os.path.exists(file_path):
		os.remove(file_path)
	np.save(file_path, data)

# here, need to give the full name of target text file (with the '.txt')
# change it for different labeled data
filename = 'labeled_data_0025_Sina.txt'
# give the name for feature file
file_name_4D = 'validation_data_4D_5x5'
file_name_3D = 'validation_data_3D_5x5'
# set the feature size, can be 1, 3 or 5
feature_size = 5
# assign the keyword for target data folder
keyword = 'SHP'


print('Loading text file...')
filepath = os.path.join(os.getcwd(), 'validation_data', filename)

with open(filepath, 'r') as f:
	data = f.readlines()

validation_data_4D = []
validation_data_3D = []

for element in data:
	target_str = element.split()
	# this command obtain the slice name and folder name
	target_path = os.path.join(os.getcwd(), os.path.basename(os.path.dirname(target_str[0])), os.path.basename(target_str[0]))
	# extract the path
	target_class = int(target_str[-1])
	# extract the class
	co_string = str(target_str[1:-1])
	result = re.findall(r"\d+\.?\d*", co_string)
	total_points = len(result) // 2
	for i in range(total_points):
		print('Current picture: ', target_path)
		# get features
		if feature_size == 3:
			feature_4D, feature_3D = features.get_assign_features_3(target_path, int(float(result[2*i])), int(float(result[2*i+1])), keyword)
		elif feature_size == 1:
			feature_4D, feature_3D = features.get_assign_features_1(target_path, int(float(result[2*i])), int(float(result[2*i+1])), keyword)
		elif feature_size == 5:
			feature_4D, feature_3D = features.get_assign_features_5(target_path, int(float(result[2*i])), int(float(result[2*i+1])), keyword)
		else:
			raise ValueError('Please give the correct size!')

		feature_4D_class = np.append(feature_4D, target_class)
		feature_3D_class = np.append(feature_3D, target_class)
		
		validation_data_4D.append(feature_4D_class)
		validation_data_3D.append(feature_3D_class)

# transfer its format
validation_data_4D = np.array(validation_data_4D)
validation_data_3D = np.array(validation_data_3D)
# save the data
save_data(file_name_4D+'.npy', validation_data_4D)
save_data(file_name_3D+'.npy', validation_data_3D)









