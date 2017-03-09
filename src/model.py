import os
import sys
import PIL
import logging
import random
import numpy as np
import read_image as ri
import matplotlib.pyplot as plt
import matplotlib.cm as cm


logger = logging.getLogger(__name__)

logging.basicConfig(filename='log_' + os.path.basename(__file__),
                            filemode='w',
                            format='%(asctime)s %(name)s %(levelname)s:\t %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)


class Model(object):

	def __init__(self, path_to_faces, num_components):
		"""
		Args
			X: the list of all images in our dataset
			ids: id numbers of all images

		Returns
			Weights of the PCA features
		"""
		print 'Training the model...'

		self.subject_count = 40
		self.train_set_count = 8
		self.test_set_count = 2
		self.img_height = 112
		self.img_width = 92

		self.img_dim = self.img_height * self.img_width
		self.path_to_faces = path_to_faces

		# We choose the number of components to be the 
		# number of images
		self.images_count = self.subject_count * self.train_set_count
		self.components_count = num_components	# we can set the number of principal
											# components here

		self.training_ids = []
		training_imgs = np.empty(shape=(self.img_dim, self.images_count), dtype='float64')

		# Should be 320 for Cambridge dataset
		logger.info('training images count' + str(self.images_count))

		curr = 0
		for i in xrange(self.subject_count):

			# grab a train_faces_count number of random face ids from this subject
			training_ids = random.sample(range(1, 11), self.train_set_count)
			
			# add this subject's list of training ids to our list of all trainining ids
			self.training_ids.append(training_ids)

			for train_id in training_ids:
				face_id = i + 1

				subj_name = 's' + str(face_id)
				img_name = str(train_id) + '.pgm'
				path_to_img = os.path.join(self.path_to_faces, subj_name, img_name)

				img = ri.read_single_image(path_to_img)
				img_vertical = np.array(img, dtype='float64').flatten()

				# Set the current column to this training image
				training_imgs[:, curr] = img_vertical[:]
				curr += 1


		# Compute the mean of all images over the columns in our matrix	
		self.mean_img = np.sum(training_imgs, axis=1) / self.images_count

		# Recenter all images
		for i in xrange(self.images_count):
			training_imgs[:, i] -= self.mean_img[:]

		# linear algebra manipulation for more efficient computation
		covar = np.matrix(training_imgs.transpose()) * np.matrix(training_imgs)
		covar /= self.images_count

		# Order the eigenvects descending by their eigenvalue
		self.eigenvals, self.eigenvects = np.linalg.eig(covar)	# eigenvects/values of the covariance matrix
		indexes = np.argsort(-self.eigenvals)					# grab indices that would sort the eigenvals desc.
		self.eigenvals = self.eigenvals[indexes]				# descending order
		self.eigenvects = self.eigenvects[indexes]				# corresponding eigenvects

		# We only use principal components corresponding to 
		# the components_count largest eigenvalues
		self.eigenvals = self.eigenvals[0:self.components_count].copy()			
		self.eigenvects = self.eigenvects[0:self.components_count].copy()

		# our eigenvects are of dim: (10304, 320). Each col reps a vector.
		self.eigenvects = self.eigenvects.transpose()									# to column form
		self.eigenvects = training_imgs * self.eigenvects	
		self.eigenvects = self.eigenvects / np.linalg.norm(self.eigenvects, axis=0)		# normalize each eigenvector

		# this represents all of the projected(!) images into the subspace??
		# (320, 10304) * (10304, 320) = (320 components, 320 images)
		self.feats = self.eigenvects.transpose() * training_imgs 	# computing the normalized weights

		# shape of our feats: (320 components, 320 images)
		logger.info("feats shape: " + str(self.eigenvects.transpose().shape) + "*" + str(training_imgs.shape) + "=" + str(self.feats.shape))
		logger.info('images dimensions' + str(np.shape(self.feats)))


	def flattenImages(self, X):
		"""
		Flattens a multi-dimensional data item into a matrix.

		Args
	     	X: List with multi-dimensional data
	    Returns
	    	A matrix that represents flattened images
		"""

		# If we have no images to process, just return an empty array
		if len(X) == 0:
			return np.array([])

		x0 = np.asarray(X[0])	# Assume that images are standardized

		# Initialize a matrix that represents the flattened images
		flat_img = np.empty((x0.size, 0), dtype=x0.dtype)

		# Grab each column in the multidimensional images 
		# and add to our horizontal matrix
		for col in X:
			flat_img = np.append(flat_img, np.asarray(col).reshape(-1,1), axis=1)
		
		# Returns a matrix that represents the images
		return np.asmatrix(flat_img)


	def classify(self, path_to_img, actual_id, k):
		"""
		Classify an image to one of the subjects using 1-NN search.

		Args
			path_to_image: path to the image to query
		Return
			id of the predicted face match
		"""
		img = ri.read_single_image(path_to_img)
		flat_img = np.array(img, dtype='float64').flatten()
		proj = self.project(flat_img)

		diff = self.feats - proj				
		distances = np.linalg.norm(diff, axis=0)	# Euclidean distance metric
		
		predicted_ids = self.kNN(distances, k)
		for predicted_id in predicted_ids:
			if (predicted_id == actual_id): return 1

		return 0


	def NN(self, distances):
		"""
		Returns the nearest match (by id) given computed distances

		Args
			distances:
				computed distances between projected image 
				and PCA subspace vectors
		"""
		return (np.argmin(distances) / self.train_set_count) + 1


	def kNN(self, distances, k):
		"""
		Returns the k nearest matches (by id) given computed distances

		Args
			distances:
				computed distances between projected image 
				and PCA subspace vectors
			k:
				number of neighbors to examine
		"""

		idx_sorted = np.array(distances).argsort()[:k]	# sorts ascending, indexes
		closest_matches = []

		for idx in idx_sorted:
			closest_matches.append((idx / self.train_set_count) + 1)

		return closest_matches


	def project(self, flat_img):
		"""
		Projects a flat image into the PCA subspace.

		Args
			a 1-dimensional array that represents an image
		Return
			reconstruction from the PCA basis: y = W_t(x - mu)
		"""

		# dims: flat image and mean image are both (10304 pixels, ) here
		flat_img -= self.mean_img
		
		# reshape the flat image so that it has a single column (10304, 1)
		# result dimension is (320 components, 1)
		return self.eigenvects.transpose() * flat_img.reshape(-1, 1)


	def reconstruct(self, projected):
		"""
		Reconstructs images
		"""

		# need to reconstruct the projection! we get (10304, 1) result
		# eigenvectors: (10304 pixels, 320 components/eigen)
		# projected: (320 components, 1 image)
		X = self.eigenvects * projected

		# add mean to each pixel in x
		# result dimension is (10304, 1)
		return X + self.mean_img.reshape(-1, 1)


	def compute(self, k=1):
		"""
		Prints the precision of the facial recognition algorithm
		on the test set.

		Uses the remaining [2] faces in each of the 40 subject folders
		as the test set to test the recognition rate.
		"""
		print 'Computing face matches initiated'

		test_count = self.test_set_count * self.subject_count

		# Should be 80 for Cambridge dataset
		logger.info('test images count' + str(self.images_count))

		test_correct = 0
		for i in xrange(self.subject_count):
			for test_id in xrange(1, 11):

				# only examine images not included in the training set
				if test_id not in self.training_ids[i]:
					
					face_id = i + 1
					subj_name = 's' + str(i + 1)
					img_name = str(test_id) + '.pgm'
					path_to_query = os.path.join(self.path_to_faces, subj_name, img_name)

					### new shit
					### classify returns a 0 or a 1: 0 for match, 1 for no match
					test_correct += self.classify(path_to_query, face_id, k)

					# predicted_id = self.classify(path_to_query, method)
					# result = (predicted_id == face_id)

					# if result == True:
					# 	test_correct += 1

		print 'Computing face matches complete'
		self.precision = float(100. * test_correct / test_count)
		print '\nCorrect: ' + str(self.precision) + '%'
		return self.precision


	def plotEigenvectors(self):
		"""
		Plots the first few eigenvectors.

		92*112 = 10304 = dimensions of image
		320 = number of training images

		eigenvectors SHAPE: (10304, 320)

			[[  2.03279215e-03   8.64818157e-03  -5.52441302e-03 ...,  -7.66921664e-04
			   -1.03883177e-02  -1.66516533e-03]
			   ...,
			[  9.61643482e-05   8.59921070e-03  -5.78722571e-03 ...,   6.75912917e-04
			  -9.47726794e-03  -1.84073129e-03]]

		features SHAPE: (320, 320)

			[[  560.25030096   866.04491486   575.8200252  ...,   665.65862941
			    527.92480344   602.42930547]
			    ...,
			[  818.50944688   922.72644718  1420.0264558  ...,   342.31421064
			  -926.41125512 -1356.09791059]]

		mean SHAPE: (10304,)
			[ 85.703125  85.63125   86.0875   ...,  77.503125  76.140625  75.728125]


		"""

		# Plots the first 16 eigenvectors
		eigenvects_to_plot = []
		for i in xrange(min(self.components_count, 16)):

			# reshape the eigenvector from (10304, 1) -> (112, 92)
			evect = self.eigenvects[:, i].reshape((self.img_height, self.img_width))
			
			# add to our list of eigenvectors to plot
			eigenvects_to_plot.append(ri.normPixel(evect))

 		self.plotImage(title='Eigenfaces', images=eigenvects_to_plot, colormap=cm.jet)


	def plotReconstruction(self, path_to_img):
		"""
		Plots the reconstruction of a face.

		Args
			path_to_img: the file path to an image
		"""
		
		img = ri.read_single_image(path_to_img)
		flat_img = np.array(img, dtype='float64').flatten()
		
		projected = self.project(flat_img)
		reconstructed = self.reconstruct(projected)
		reconstructed = reconstructed.reshape((self.img_height, self.img_width))
		
		to_plot = [ri.normPixel(reconstructed)]
		self.plotImage(title='Reconstruction', images=to_plot)


	def plotImage(self, title, images, colormap=cm.gray):
		"""
		Saves an image in the current directory given a list of image arrays.
		"""
		fig = plt.figure()
		for i in xrange(len(images)):
		 	plt.imshow(np.asarray(images[i]), cmap=colormap)
			fig.savefig(title + str(i))


	def create_font(self, fontname='Tahoma', fontsize=10):
	    return { 'fontname': fontname, 'fontsize':fontsize }




