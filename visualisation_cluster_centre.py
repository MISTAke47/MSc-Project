'''
This file will visualise the centre of the cluster as grayscale image
used for interpretation

Author: Yan Gao
email: gaoy4477@gmail.com
'''
import os
import numpy as np 
from joblib import load
import argparse
import matplotlib.pyplot as plt

# these two lines to solve the bug reported on Mojave 10.14
import matplotlib
matplotlib.use('MacOSX')

def get_args():
	parser = argparse.ArgumentParser(description='visualise model centre')
	parser.add_argument('--model_4D', nargs="?", type=str, 
                        help='File name of saved model for 4D data')
	parser.add_argument('--model_3D', nargs="?", type=str, 
                        help='File name of saved model for 3D data')
	parser.add_argument('--type', nargs="?", type=str,
						help='Type of clustering algorithm')
	args = parser.parse_args()
	print(args)
	return args

# this function will transfer 9x1 array to 3x3 array with uint16 type
def get_image(array):
	dim = int(np.sqrt(len(array)))
	convert = array.astype(np.uint16)
	img = convert.reshape((dim, dim))
	return img



args = get_args()

model_4D_path = os.path.join(os.getcwd(), 'model', args.model_4D+'.model')
model_3D_path = os.path.join(os.getcwd(), 'model', args.model_3D+'.model')

model_4D_type = load(model_4D_path)
model_3D_type = load(model_3D_path)

if args.type == 'kmeans' or 'mini_batch_kmeans':
	centre_4D = model_4D_type.cluster_centers_
	centre_3D = model_3D_type.cluster_centers_
# elif args.type == 'gmm':
# 	continue
else:
	raise ValueError('Please input the correct type.')

num_centres_4D = centre_4D.shape[0]
num_centres_3D = centre_3D.shape[0]

dim_4D = centre_4D[0].shape[0]
dim_3D = centre_3D[0].shape[0]

# This part show the centre for 3D data
size_img = int(dim_3D/3)
for i in range(num_centres_3D):
# loop for all centres
	array_1 = centre_3D[i][:size_img]
	img_1 = get_image(array_1)
	# mid slice
	array_2 = centre_3D[i][size_img:2*size_img]
	img_2 = get_image(array_2)
	# bot slice
	array_3 = centre_3D[i][2*size_img:]
	img_3 = get_image(array_3)
	# top slice

	# plot the centre
	fig = plt.figure()
	fig.suptitle('Visualisation for 3D centre {:d}'.format(i+1), x=0.5, y=0.8)
	ax = plt.subplot(131)
	ax.imshow(img_1, 'gray')
	ax.set_title('Bot slice')
	ax = plt.subplot(132)
	ax.imshow(img_2, 'gray')
	ax.set_title('Mid slice')
	ax = plt.subplot(133)
	ax.imshow(img_3, 'gray')
	ax.set_title('Top slice')
	name = '3D_centre_'+str(i+1)+'.png'
	fig.savefig(name, bbox_inches='tight', pad_inches=0.0)

# This part show the centre for 4D data
size_img = int(dim_4D/3/3)
size_t = int(dim_4D/3)
for i in range(num_centres_4D):
	# there are 3 timestamps for 4D data
	# image 1-3 for current timestamp
	array_1 = centre_4D[i][:size_t][:size_img]
	img_1 = get_image(array_1)
	# mid slice
	array_2 = centre_4D[i][:size_t][size_img:2*size_img]
	img_2 = get_image(array_2)
	# bot slice
	array_3 = centre_4D[i][:size_t][2*size_img:]
	img_3 = get_image(array_3)

	# img 4-6 for previous timestamp
	array_4 = centre_4D[i][size_t:2*size_t][:size_img]
	img_4 = get_image(array_4)
	# mid slice
	array_5 = centre_4D[i][size_t:2*size_t][size_img:2*size_img]
	img_5 = get_image(array_5)
	# bot slice
	array_6 = centre_4D[i][size_t:2*size_t][2*size_img:]
	img_6 = get_image(array_6)

	# img 7-9 for next timestamp
	array_7 = centre_4D[i][2*size_t:][:size_img]
	img_7 = get_image(array_7)
	# mid slice
	array_8 = centre_4D[i][2*size_t:][size_img:2*size_img]
	img_8 = get_image(array_8)
	# bot slice
	array_9 = centre_4D[i][2*size_t:][2*size_img:]
	img_9 = get_image(array_9)

	# plot the centre
	fig = plt.figure()
	fig.suptitle('Visualisation for 4D centre {:d}'.format(i+1)) 
	ax = plt.subplot(331)
	ax.set_title('Bot slice')
	ax.imshow(img_1, 'gray')
	ax = plt.subplot(332)
	ax.set_title('Mid slice')
	ax.imshow(img_2, 'gray')
	ax = plt.subplot(333)
	ax.set_title('Top slice')
	ax.imshow(img_3, 'gray')
	
	ax = plt.subplot(334)
	ax.imshow(img_4, 'gray')
	# ax.set_title('bot slice')
	ax = plt.subplot(335)
	ax.imshow(img_5, 'gray')
	# ax.set_title('mid slice')
	ax = plt.subplot(336)
	ax.imshow(img_6, 'gray')
	# ax.set_title('top slice')

	ax = plt.subplot(337)
	ax.imshow(img_7, 'gray')
	# ax.set_title('bot slice')
	ax = plt.subplot(338)
	ax.imshow(img_8, 'gray')
	# ax.set_title('mid slice')
	ax = plt.subplot(339)
	ax.imshow(img_9, 'gray')
	# ax.set_title('top slice')

	name = '4D_centre_'+str(i+1)+'.png'
	fig.savefig(name, bbox_inches='tight', pad_inches=0.0)

plt.show()






