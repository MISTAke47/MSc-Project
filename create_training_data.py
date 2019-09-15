'''
This file return the feature for training points

To running this file, we need to assign the parameter of the mask (a circle area), which can be obtained by running find_mask.py

We need to set the subsampling rate, range of target timestamp and target slice to reduce the data size

Author: Yan Gao
email: gaoy4477@gmail.com
'''
import os
import numpy as np 
import random
import argparse
import module.content as content
import module.features as features

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(description='Feature extraction')

    parser.add_argument('--begin_timestamp', nargs="?", type=str, default='0025',
                        help='Sample begin from this timestamp')
    parser.add_argument('--end_timestamp', nargs="?", type=str, default='0025',
                        help='Sample end at this timestamp')
    parser.add_argument('--begin_slice', nargs="?", type=int, default=600, 
                        help='Sample begin from this timestamp')
    parser.add_argument('--end_slice', nargs="?", type=int, default=800, 
                        help='Sample end at this timestamp')
    parser.add_argument('--filename_3D', nargs="?", type=str, default="training_data_3D",
                        help='File name of saved 3D feature')
    parser.add_argument('--filename_4D', nargs="?", type=str, default="training_data_4D",
                        help='File name of saved 4D feature')
    parser.add_argument('--size', nargs="?", type=int, default=3,
                        help='Type of different size of the area')

    args = parser.parse_args()
    print(args)
    return args

# set parameters
# set the parameter of mask here
mask_centre = (700, 810)
radius = 550
# for locate target folder
keyword = 'SHP'
# set the subsampling rate here
# sub sampling. We suggest set to make the total numpber of points being 2 millon
subsampling_rate = 0.01


# get argument from args function
args = get_args()
current_path = os.getcwd()
all_timestamp = content.get_folder(current_path, keyword)
# get the index for begin and end
begin_timestamp_index = [all_timestamp.index(i) for i in all_timestamp if args.begin_timestamp in i]
end_timestamp_index = [all_timestamp.index(i) for i in all_timestamp if args.end_timestamp in i]
# create target timestamp list
target_timestamp = all_timestamp[begin_timestamp_index[0]: end_timestamp_index[0]+1]

# create the './training_data' to save the data
save_folder = os.path.join(current_path,'training_data')
if not os.path.exists(save_folder):
	os.mkdir(save_folder)


all_feature_4D = []
all_feature_3D = []

for sub_timestamp in target_timestamp:
	sub_path = os.path.join(current_path, sub_timestamp)
	print('Current timestamp:', sub_path)
	sub_all_tif = content.get_allslice(sub_path)
	target_slice = sub_all_tif[args.begin_slice-1: args.end_slice]

	# use feature_index to get the features
	_, feature_index = features.get_mask(sub_all_tif[0], mask_centre, radius, args.size)
	total_feature_num = len(feature_index)
	sample_feature_num = int(subsampling_rate * total_feature_num)
	sample_list = random.sample(range(total_feature_num), sample_feature_num)
	# get the sample index
	sub_feature_index = [feature_index[i] for i in sample_list]
	
	for each_tif in target_slice:
		if args.size == 3:
			feature_4D, feature_3D = features.get_all_features_3(each_tif, sub_feature_index, keyword)
		elif args.size == 1:
			feature_4D, feature_3D = features.get_all_features_1(each_tif, sub_feature_index, keyword)
		elif args.size == 5:
			feature_4D, feature_3D = features.get_all_features_5(each_tif, sub_feature_index, keyword)
		else:
			raise ValueError('Please set the right size type!')

		all_feature_4D.append(feature_4D)
		all_feature_3D.append(feature_3D)

print('Collected the data!')

print('Saving data...')
training_data_4D = np.array(all_feature_4D)
training_data_3D = np.array(all_feature_3D)

# file format should be .npy
file_path_4D = os.path.join(save_folder, args.filename_4D + '.npy')
file_path_3D = os.path.join(save_folder, args.filename_3D + '.npy')

# before saving the data, delete the data with same name
if os.path.exists(file_path_4D):
	os.remove(file_path_4D)
if os.path.exists(file_path_3D):
	os.remove(file_path_3D)

np.save(file_path_4D, training_data_4D)
np.save(file_path_3D, training_data_3D)
print('Finished!')



