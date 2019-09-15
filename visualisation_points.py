'''
This file will visualise the average grayscale value for pore and non-pore point
load data from validation data
used for interpretation

Author: Yan Gao
email: gaoy4477@gmail.com
'''
import re
import numpy as np 
import os
import module.features as features
import matplotlib.pyplot as plt 

# these two lines to solve the bug reported on Mojave 10.14
import matplotlib
matplotlib.use('MacOSX')

# this function will transfer 9x1 array to 3x3 array with uint16 type
def get_image(array):
	dim = int(np.sqrt(len(array)))
	convert = array.astype(np.uint16)
	img = convert.reshape((dim, dim))
	return img

# transfer a 27-dim array to 3 images
def transfer_image(array):
	length = int(len(array))
	div_length = length // 3
	img_bot = get_image(array[:div_length])
	img_mid = get_image(array[div_length:2*div_length])
	img_top = get_image(array[2*div_length:])
	return img_bot, img_mid, img_top


filename_4D = 'validation_data_4D_3x3'
filename_3D = 'validation_data_3D_3x3'


# This part we load validation data
filepath_4D = os.path.join(os.getcwd(), 'validation_data', filename_4D+'.npy')
filepath_3D = os.path.join(os.getcwd(), 'validation_data', filename_3D+'.npy')

validation_data_4D = np.load(filepath_4D)
validation_data_3D = np.load(filepath_3D)


# devide pore and non-pore
data_label = validation_data_4D[:, -1]
pore_index =np.argwhere(data_label==0)
non_pore_index = np.argwhere(data_label==1)
artifact_index = np.argwhere(data_label==2)
num_pore = len(pore_index)
num_non_pore = len(non_pore_index)
num_artifact = len(artifact_index)

validation_data_4D_pore = np.zeros((num_pore, validation_data_4D.shape[1]))
validation_data_4D_non_pore = np.zeros((num_non_pore, validation_data_4D.shape[1]))
validation_data_4D_artifact = np.zeros((num_artifact, validation_data_4D.shape[1]))
validation_data_3D_pore = np.zeros((num_pore, validation_data_3D.shape[1]))
validation_data_3D_non_pore = np.zeros((num_non_pore, validation_data_3D.shape[1]))
validation_data_3D_artifact = np.zeros((num_artifact, validation_data_3D.shape[1]))

for index, i in enumerate(pore_index):
    validation_data_4D_pore[index] = validation_data_4D[i]
    validation_data_3D_pore[index] = validation_data_3D[i]
    
for index, i in enumerate(non_pore_index):
    validation_data_4D_non_pore[index] = validation_data_4D[i]
    validation_data_3D_non_pore[index] = validation_data_3D[i]

for index, i in enumerate(artifact_index):
    validation_data_4D_artifact[index] = validation_data_4D[i]
    validation_data_3D_artifact[index] = validation_data_3D[i]

pore_mean_3D = np.mean(validation_data_3D_pore, axis=0)[:-1]
non_pore_mean_3D = np.mean(validation_data_3D_non_pore, axis=0)[:-1]
artifact_mean_3D = np.mean(validation_data_3D_artifact, axis=0)[:-1]

pore_mean_4D = np.mean(validation_data_4D_pore, axis=0)[:-1]
non_pore_mean_4D = np.mean(validation_data_4D_non_pore, axis=0)[:-1]
artifact_mean_4D = np.mean(validation_data_4D_artifact, axis=0)[:-1]


img_bot_pore_3D, img_mid_pore_3D, img_top_pore_3D = transfer_image(pore_mean_3D)
img_bot_nonpore_3D, img_mid_nonpore_3D, img_top_nonpore_3D = transfer_image(non_pore_mean_3D)
img_bot_artifact_3D, img_mid_artifact_3D, img_top_artifact_3D = transfer_image(artifact_mean_3D)

img_bot_pore_4D_t0, img_mid_pore_4D_t0, img_top_pore_4D_t0 = transfer_image(pore_mean_4D[:27])
img_bot_nonpore_4D_t0, img_mid_nonpore_4D_t0, img_top_nonpore_4D_t0 = transfer_image(non_pore_mean_4D[:27])
img_bot_artifact_4D_t0, img_mid_artifact_4D_t0, img_top_artifact_4D_t0 = transfer_image(artifact_mean_4D[:27])


img_bot_pore_4D_t1, img_mid_pore_4D_t1, img_top_pore_4D_t1 = transfer_image(pore_mean_4D[27:54])
img_bot_nonpore_4D_t1, img_mid_nonpore_4D_t1, img_top_nonpore_4D_t1 = transfer_image(non_pore_mean_4D[27:54])
img_bot_artifact_4D_t1, img_mid_artifact_4D_t1, img_top_artifact_4D_t1 = transfer_image(artifact_mean_4D[27:54])

img_bot_pore_4D_t2, img_mid_pore_4D_t2, img_top_pore_4D_t2 = transfer_image(pore_mean_4D[54:])
img_bot_nonpore_4D_t2, img_mid_nonpore_4D_t2, img_top_nonpore_4D_t2 = transfer_image(non_pore_mean_4D[54:])
img_bot_artifact_4D_t2, img_mid_artifact_4D_t2, img_top_artifact_4D_t2 = transfer_image(artifact_mean_4D[54:])



fig = plt.figure()
fig.suptitle('Visualisation for pore in 3D space', x=0.5, y=0.8)
ax = plt.subplot(131)
# ax.axis('off')
ax.set_title('Bot slice')
ax.imshow(img_bot_pore_3D,'gray')
ax = plt.subplot(132)
# ax.axis('off')
ax.set_title('Mid slice')
ax.imshow(img_mid_pore_3D,'gray')
ax = plt.subplot(133)
# ax.axis('off')
ax.set_title('Top slice')
ax.imshow(img_top_pore_3D,'gray')
fig.savefig('3D_pore.png', bbox_inches='tight', pad_inches=0.0)

fig = plt.figure()
fig.suptitle('Visualisation for non-pore (normal) in 3D space', x=0.5, y=0.8)
ax = plt.subplot(131)
# ax.axis('off')
ax.set_title('Bot slice')
ax.imshow(img_bot_nonpore_3D,'gray')
ax = plt.subplot(132)
# ax.axis('off')
ax.set_title('Mid slice')
ax.imshow(img_mid_nonpore_3D,'gray')
ax = plt.subplot(133)
# ax.axis('off')
ax.set_title('Top slice')
ax.imshow(img_top_nonpore_3D,'gray')
fig.savefig('3D_nonpore.png', bbox_inches='tight', pad_inches=0.0)

fig = plt.figure()
fig.suptitle('Visualisation for non-pore (artifact) in 3D space', x=0.5, y=0.8)
ax = plt.subplot(131)
# ax.axis('off')
ax.set_title('Bot slice')
ax.imshow(img_bot_artifact_3D,'gray')
ax = plt.subplot(132)
# ax.axis('off')
ax.set_title('Mid slice')
ax.imshow(img_mid_artifact_3D,'gray')
ax = plt.subplot(133)
# ax.axis('off')
ax.set_title('Top slice')
ax.imshow(img_top_artifact_3D,'gray')
fig.savefig('3D_artifact.png', bbox_inches='tight', pad_inches=0.0)

fig = plt.figure()
fig.suptitle('Visualisation for pore in 4D space')
ax = plt.subplot(331)
# ax.axis('off')
ax.set_title('Bot slice')
ax.imshow(img_bot_pore_4D_t0,'gray')
ax = plt.subplot(332)
# ax.axis('off')
ax.set_title('Mid slice')
ax.imshow(img_mid_pore_4D_t0,'gray')
ax = plt.subplot(333)
# ax.axis('off')
ax.set_title('Top slice')
ax.imshow(img_top_pore_4D_t0,'gray')
ax = plt.subplot(334)
# ax.axis('off')
ax.imshow(img_bot_pore_4D_t1,'gray')
ax = plt.subplot(335)
# ax.axis('off')
ax.imshow(img_mid_pore_4D_t1,'gray')
ax = plt.subplot(336)
# ax.axis('off')
ax.imshow(img_top_pore_4D_t1,'gray')
ax = plt.subplot(337)
# ax.axis('off')
ax.imshow(img_bot_pore_4D_t2,'gray')
ax = plt.subplot(338)
# ax.axis('off')
ax.imshow(img_mid_pore_4D_t2,'gray')
ax = plt.subplot(339)
# ax.axis('off')
ax.imshow(img_top_pore_4D_t2,'gray')
fig.savefig('4D_pore.png', bbox_inches='tight', pad_inches=0.0)


fig = plt.figure()
fig.suptitle('Visualisation for non-pore (normal) in 4D space')
ax = plt.subplot(331)
# ax.axis('off')
ax.set_title('Bot slice')
ax.imshow(img_bot_nonpore_4D_t0,'gray')
ax = plt.subplot(332)
# ax.axis('off')
ax.set_title('Mid slice')
ax.imshow(img_mid_nonpore_4D_t0,'gray')
ax = plt.subplot(333)
# ax.axis('off')
ax.set_title('Top slice')
ax.imshow(img_top_nonpore_4D_t0,'gray')
ax = plt.subplot(334)
# ax.axis('off')
ax.imshow(img_bot_nonpore_4D_t1,'gray')
ax = plt.subplot(335)
# ax.axis('off')
ax.imshow(img_mid_nonpore_4D_t1,'gray')
ax = plt.subplot(336)
# ax.axis('off')
ax.imshow(img_top_nonpore_4D_t1,'gray')
ax = plt.subplot(337)
# ax.axis('off')
ax.imshow(img_bot_nonpore_4D_t2,'gray')
ax = plt.subplot(338)
# ax.axis('off')
ax.imshow(img_mid_nonpore_4D_t2,'gray')
ax = plt.subplot(339)
# ax.axis('off')
ax.imshow(img_top_nonpore_4D_t2,'gray')
fig.savefig('4D_nonpore.png', bbox_inches='tight', pad_inches=0.0)

fig = plt.figure()
fig.suptitle('Visualisation for non-pore (artifact) in 4D space')
ax = plt.subplot(331)
# ax.axis('off')
ax.set_title('Bot slice')
ax.imshow(img_bot_artifact_4D_t0,'gray')
ax = plt.subplot(332)
# ax.axis('off')
ax.set_title('Mid slice')
ax.imshow(img_mid_artifact_4D_t0,'gray')
ax = plt.subplot(333)
# ax.axis('off')
ax.set_title('Top slice')
ax.imshow(img_top_artifact_4D_t0,'gray')
ax = plt.subplot(334)
# ax.axis('off')
ax.imshow(img_bot_artifact_4D_t1,'gray')
ax = plt.subplot(335)
# ax.axis('off')
ax.imshow(img_mid_artifact_4D_t1,'gray')
ax = plt.subplot(336)
# ax.axis('off')
ax.imshow(img_top_artifact_4D_t1,'gray')
ax = plt.subplot(337)
# ax.axis('off')
ax.imshow(img_bot_artifact_4D_t2,'gray')
ax = plt.subplot(338)
# ax.axis('off')
ax.imshow(img_mid_artifact_4D_t2,'gray')
ax = plt.subplot(339)
# ax.axis('off')
ax.imshow(img_top_artifact_4D_t2,'gray')
fig.savefig('4D_artifact.png', bbox_inches='tight', pad_inches=0.0)

plt.show()










