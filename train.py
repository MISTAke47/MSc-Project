'''
This file will do the training process.
It will automatically load and concatenate traning data from "traning_data" folder
It will return the model as "name.model" in "model" folder


Author: Yan Gao
email: gaoy4477@gmail.com
'''
import os
import glob
import numpy as np 
import module.train as train
import argparse

def get_args():
	parser = argparse.ArgumentParser(description='Training process')
	parser.add_argument('--data_3D', nargs="?", type=str, 
    					help='File name of saved 3D feature')
	parser.add_argument('--data_4D', nargs="?", type=str, 
                        help='File name of saved 4D feature')
	parser.add_argument('--model_type', nargs="?", type=str, 
                        help='Type of model, shoulde be kmeans, mini_batch_kmeans or gmm')
	parser.add_argument('--num_cluster', nargs="?", type=int, default=2,
    					help='Number of clusters')
	parser.add_argument('--covariance_type', nargs="?", type=str, default='None',
    					help='Type of covariance for gmm model, if not use gmm, do not need to assign it')
	parser.add_argument('--name_4D', nargs="?", type=str, default='none',
                        help='File name of saved model for 4D data')
	parser.add_argument('--name_3D', nargs="?", type=str, default='none',
                        help='File name of saved model for 3D data')
	args = parser.parse_args()
	print(args)
	return args


args = get_args()

print('Loading data...')
training_data_4D_path = os.path.join(os.getcwd(), 'training_data', args.data_4D+'.npy')
training_data_3D_path = os.path.join(os.getcwd(), 'training_data', args.data_3D+'.npy')

training_data_4D = train.load_training_data(training_data_4D_path)
training_data_3D = train.load_training_data(training_data_3D_path)
print('Finished!')

print('Number of samples:', training_data_4D.shape[0])

save_folder = os.path.join(os.getcwd(), 'model')
if not os.path.exists(save_folder):
	os.mkdir(save_folder)

model_4D_path = os.path.join(save_folder, args.name_4D + '.model')
model_3D_path = os.path.join(save_folder, args.name_3D + '.model')

if args.model_type == 'kmeans':
	# kmeans 
	# train.kmeans_algorithm(num_cluster, training_data, filename)
	if args.name_4D != 'none':
		print('kmeans for 4D data')
		train.kmeans_algorithm(args.num_cluster, training_data_4D, model_4D_path)
	
	if args.name_3D != 'none':
		print('kmeans for 3D data')
		train.kmeans_algorithm(args.num_cluster, training_data_3D, model_3D_path)
elif args.model_type == 'mini_batch_kmeans':
	# mini-batch kmeans
	# train.mini_batch_kmeans_algorithm(num_cluster, training_data, filename)
	if args.name_4D != 'none':
		print('mini-batch kmeans for 4D data')
		train.mini_batch_kmeans_algorithm(args.num_cluster, training_data_4D, model_4D_path)
	
	if args.name_3D != 'none':
		print('mini-batch kmeans for 3D data')
		train.mini_batch_kmeans_algorithm(args.num_cluster, training_data_3D, model_3D_path)
elif args.model_type == 'gmm':
	# GMM
	# train.gmm(num_components, covariance_type, training_data, filename)
	# There are four covariance_type: {‘full’, ‘tied’, ‘diag’, ‘spherical’}
	# all the parameters need to be assigned
	if args.name_4D != 'none':
		print('GMM for 4D data')
		train.gmm(args.num_cluster, args.covariance_type, training_data_4D, model_4D_path)
	
	if args.name_3D != 'none':
		print('GMM for 3D data')
		train.gmm(args.num_cluster, args.covariance_type, training_data_3D, model_3D_path)
elif args.model_type == 'mean_shift':
	# all data is too large for mean shift, just use first 20,000 samples
	num_subsample = 10000

	if args.name_4D != 'none':
		print('Mean shift for 4D data')
		print('Shuffle data and use first {:d}'.format(num_subsample))
		bandwidth_4D = 30000
		np.random.shuffle(training_data_4D)
		training_data_4D = training_data_4D[:num_subsample]
		train.mean_shift(training_data_4D, model_4D_path, bandwidth_4D)

	if args.name_3D != 'none':
		print('Mean shift for 3D data')
		print('Shuffle data and use first {:d}'.format(num_subsample))
		bandwidth_3D = 10000
		np.random.shuffle(training_data_3D)
		training_data_3D = training_data_3D[:num_subsample]
		train.mean_shift(training_data_3D, model_3D_path, bandwidth_3D)

# drop DBSCAN: cannot generalize to other data
# elif args.model_type == 'dbscan':
# 	print('DBSCAN for 4D data')
# 	train.dbscan(training_data_4D, model_4D_path)
# 	print('DBSCAN for 3D data')
# 	train.dbscan(training_data_3D, model_3D_path)
else:
	raise ValueError('Please input the correct type name!')








