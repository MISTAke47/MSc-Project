'''
This file will return the visualisation based on UMAP
If we assign the centre from kmeans, it can show the location of the centre in the transformed space
used for interpretation

Author: Yan Gao
email: gaoy4477@gmail.com
'''
import os
import numpy as np 
import umap
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import module.train as train
from joblib import load

# ignore the warning from umap
import warnings

# these two lines to solve the bug reported on Mojave 10.14
import matplotlib
matplotlib.use('MacOSX')

# will ignore some warnings
warnings.filterwarnings('ignore')

# if dont want to show the centre, just assign flag_show_centre = False
# show centre only for kmeans or mini_batch_kmeans algorithm
flag_show_centre = True
model_4D = 'mini_kmeans_4D_3_3x3_0025'
model_3D = 'mini_kmeans_3D_3_3x3_0025'
# Load the data
data_3D = 'training_data_3D_3x3_0025'
data_4D = 'training_data_4D_3x3_0025'
# assign the pore, this should match the model
pore_4D = 0
pore_3D = 0

# load artifact points
name_4D = 'artifact_4D_3x3'
name_3D = 'artifact_3D_3x3'
path_4D = os.path.join(os.getcwd(), 'validation_data', name_4D+'.npy')
path_3D = os.path.join(os.getcwd(), 'validation_data', name_3D+'.npy')

artifact_data_4D = np.load(path_4D)
artifact_data_3D = np.load(path_3D)

artifact_4D = artifact_data_4D[:, :-1]
artifact_3D = artifact_data_3D[:, :-1]


# load pore points
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
num_pore = len(pore_index)
num_non_pore = len(non_pore_index)

validation_data_4D_pore = np.zeros((num_pore, validation_data_4D.shape[1]))
validation_data_4D_non_pore = np.zeros((num_non_pore, validation_data_4D.shape[1]))
validation_data_3D_pore = np.zeros((num_pore, validation_data_3D.shape[1]))
validation_data_3D_non_pore = np.zeros((num_non_pore, validation_data_3D.shape[1]))

for index, i in enumerate(pore_index):
    validation_data_4D_pore[index] = validation_data_4D[i]
    validation_data_3D_pore[index] = validation_data_3D[i]
    
for index, i in enumerate(non_pore_index):
    validation_data_4D_non_pore[index] = validation_data_4D[i]
    validation_data_3D_non_pore[index] = validation_data_3D[i]



data_4D_path = os.path.join(os.getcwd(), 'training_data', data_4D+'.npy')
data_3D_path = os.path.join(os.getcwd(), 'training_data', data_3D+'.npy')

print('Loading data...')
training_data_4D = train.load_training_data(data_4D_path)
training_data_3D = train.load_training_data(data_3D_path)

print('Shuffling the data...')
np.random.shuffle(training_data_4D)
np.random.shuffle(training_data_3D)

num_points = 50000
print('Using subset of the data, {:d} points'.format(num_points))
subset_training_data_4D = training_data_4D[:num_points]
subset_training_data_3D = training_data_3D[:num_points]

print('Embedding for 4D data...')
# embedding_4D_2_model = umap.UMAP(n_components=2).fit(subset_training_data_4D)
embedding_4D_3_model = umap.UMAP(n_components=3).fit(subset_training_data_4D)
print('Finished!')
print('Embedding for 3D data...')
# embedding_3D_2_model = umap.UMAP(n_components=2).fit(subset_training_data_3D)
embedding_3D_3_model = umap.UMAP(n_components=3).fit(subset_training_data_3D)
print('Finished!')

# if want to show the centre, will run following code
if flag_show_centre:
	model_4D_path = os.path.join(os.getcwd(), 'model', model_4D+'.model')
	model_3D_path = os.path.join(os.getcwd(), 'model', model_3D+'.model')

	model_4D_type = load(model_4D_path)
	model_3D_type = load(model_3D_path)

	centre_4D = model_4D_type.cluster_centers_
	centre_3D = model_3D_type.cluster_centers_
	
	num_centre_4D = centre_4D.shape[0]
	num_centre_3D = centre_3D.shape[0]

	# Assume we have known just 2 centres, we can change this assumption
	# project 4D centre
	centre_4D_list_3D = [embedding_4D_3_model.transform(centre_4D[i].reshape(1,-1)) for i in range(num_centre_4D)]
	# centre_4D_list_2D = [embedding_4D_2_model.transform(centre_4D[i].reshape(1,-1)) for i in range(num_centre_4D)]

	centre_3D_list_3D = [embedding_3D_3_model.transform(centre_3D[i].reshape(1,-1)) for i in range(num_centre_3D)]
	# centre_3D_list_2D = [embedding_3D_2_model.transform(centre_3D[i].reshape(1,-1)) for i in range(num_centre_3D)]


print('Transforming the data...')
# transform the data
# embedding_4D_2 = embedding_4D_2_model.transform(subset_training_data_4D)
embedding_4D_3 = embedding_4D_3_model.transform(subset_training_data_4D)
# embedding_3D_2 = embedding_3D_2_model.transform(subset_training_data_3D)
embedding_3D_3 = embedding_3D_3_model.transform(subset_training_data_3D)

embedding_pore_4D = embedding_4D_3_model.transform(validation_data_4D_pore)
embedding_non_pore_4D = embedding_4D_3_model.transform(validation_data_4D_non_pore)
embedding_artifact_4D = embedding_4D_3_model.transform(artifact_4D)

embedding_pore_3D = embedding_3D_3_model.transform(validation_data_3D_pore)
embedding_non_pore_3D = embedding_3D_3_model.transform(validation_data_3D_non_pore)
embedding_artifact_3D = embedding_3D_3_model.transform(artifact_3D)




print('Plotting...')
plt.figure()
ax = plt.subplot(projection='3d')
ax.scatter(embedding_4D_3[:,0], embedding_4D_3[:,1], embedding_4D_3[:,2], alpha=0.01, color='yellow')
if flag_show_centre:
	for i in range(num_centre_4D):
		if i == pore_4D:
			ax.scatter(centre_4D_list_3D[i][:,0], centre_4D_list_3D[i][:,1], centre_4D_list_3D[i][:,2], color='red')
		else:
			ax.scatter(centre_4D_list_3D[i][:,0], centre_4D_list_3D[i][:,1], centre_4D_list_3D[i][:,2], color='blue')
ax.set_title('3D projection for 4D data',fontsize=12,color='r')

plt.figure()
ax = plt.subplot(projection='3d')
ax.scatter(embedding_3D_3[:,0], embedding_3D_3[:,1], embedding_3D_3[:,2], alpha=0.01, color='yellow')
if flag_show_centre:
	for i in range(num_centre_3D):
		if i == pore_3D:
			ax.scatter(centre_3D_list_3D[i][:,0], centre_3D_list_3D[i][:,1], centre_3D_list_3D[i][:,2], color='red')
		else:
			ax.scatter(centre_3D_list_3D[i][:,0], centre_3D_list_3D[i][:,1], centre_3D_list_3D[i][:,2], color='blue')
ax.set_title('3D projection for 3D data',fontsize=12,color='r')



plt.figure()
ax = plt.subplot(projection='3d')
ax.scatter(embedding_pore_4D[:,0], embedding_pore_4D[:,1], embedding_pore_4D[:,2], alpha=0.2, color='orange')
ax.scatter(embedding_non_pore_4D[:,0], embedding_non_pore_4D[:,1], embedding_non_pore_4D[:,2], alpha=0.2, color='violet')
ax.scatter(embedding_artifact_4D[:,0], embedding_artifact_4D[:,1], embedding_artifact_4D[:,2], alpha=0.2, color='green')
if flag_show_centre:
	for i in range(num_centre_4D):
		if i == pore_4D:
			ax.scatter(centre_4D_list_3D[i][:,0], centre_4D_list_3D[i][:,1], centre_4D_list_3D[i][:,2], color='red')
		else:
			ax.scatter(centre_4D_list_3D[i][:,0], centre_4D_list_3D[i][:,1], centre_4D_list_3D[i][:,2], color='blue')
ax.set_title('3D projection for 4D data',fontsize=12,color='r')


plt.figure()
ax = plt.subplot(projection='3d')
ax.scatter(embedding_pore_3D[:,0], embedding_pore_3D[:,1], embedding_pore_3D[:,2], alpha=0.2, color='orange')
ax.scatter(embedding_non_pore_3D[:,0], embedding_non_pore_3D[:,1], embedding_non_pore_3D[:,2], alpha=0.2, color='violet')
ax.scatter(embedding_artifact_3D[:,0], embedding_artifact_3D[:,1], embedding_artifact_3D[:,2], alpha=0.2, color='green')
if flag_show_centre:
	for i in range(num_centre_3D):
		if i == pore_3D:
			ax.scatter(centre_3D_list_3D[i][:,0], centre_3D_list_3D[i][:,1], centre_3D_list_3D[i][:,2], color='red')
		else:
			ax.scatter(centre_3D_list_3D[i][:,0], centre_3D_list_3D[i][:,1], centre_3D_list_3D[i][:,2], color='blue')
ax.set_title('3D projection for 3D data',fontsize=12,color='r')


# plt.figure()
# ax = plt.subplot(211)
# ax.scatter(embedding_4D_2[:,0],embedding_4D_2[:,1])
# if flag_show_centre:
# 	for i in range(num_centre_4D):
# 		ax.scatter(centre_4D_list_2D[i][:,0], centre_4D_list_2D[i][:,1], color='red')
# ax.set_title('2D projection for 4D data',fontsize=12,color='r')
# ax = plt.subplot(212)
# ax.scatter(embedding_3D_2[:,0],embedding_3D_2[:,1])
# if flag_show_centre:
# 	for i in range(num_centre_3D):
# 		ax.scatter(centre_3D_list_2D[i][:,0], centre_3D_list_2D[i][:,1], color='red')
# ax.set_title('2D projection for 3D data',fontsize=12,color='r')


plt.show()







