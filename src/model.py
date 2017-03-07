import os
import sys
import PIL
import logging
import random
import numpy as np
import read_image as ri


logger = logging.getLogger(__name__)

logging.basicConfig(filename='log_' + os.path.basename(__file__),
                            filemode='w',
                            format='%(asctime)s %(name)s %(levelname)s:\t %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)


class Model(object):

	def __init__(self, path_to_faces):
		"""
		Args
			X: the list of all images in our dataset
			ids: id numbers of all images

		Returns
			Weights of the PCA features
		"""
		print 'Training the model...'

		self.faces_count = 40
		self.train_faces_count = 8
		self.test_faces_count = 2
		self.img_dim = 92 * 112
		self.path_to_faces = path_to_faces

		# We choose the number of components to be the 
		# number of images
		self.images_ct = self.faces_count * self.train_faces_count
		self.components_ct = self.images_ct	# we can set the number of principal
											# components here

		self.training_ids = []
		self.training_imgs = np.empty(shape=(self.img_dim, self.images_ct), dtype='float64')

		# Should be 320 for Cambridge dataset
		logger.info('training images count' + str(self.images_ct))

		curr = 0
		for i in xrange(self.faces_count):

			# grab a train_faces_ct number of random face ids from this subject
			training_ids = random.sample(range(1, 11), self.train_faces_count)
			
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
				self.training_imgs[:, curr] = img_vertical[:]
				curr += 1


		# Compute the mean of all images over the columns in our matrix	
		self.mean_img = np.sum(self.training_imgs, axis=1) / self.images_ct

		# Recenter all images
		for i in xrange(self.images_ct):
			self.training_imgs[:, i] -= self.mean_img[:]

		# linear algebra manipulation for more efficient computation
		covar = np.matrix(self.training_imgs.transpose()) * np.matrix(self.training_imgs)
		covar /= self.images_ct

		# Order the eigenvects descending by their eigenvalue
		self.eigenvals, self.eigenvects = np.linalg.eig(covar)	# eigenvects/values of the covariance matrix
		indexes = np.argsort(-self.eigenvals)					# grab indices that would sort the eigenvals desc.
		self.eigenvals = self.eigenvals[indexes]				# descending order
		self.eigenvects = self.eigenvects[indexes]				# corresponding eigenvects

		# We only use principal components corresponding to 
		# the components_ct largest eigenvalues
		self.eigenvals = self.eigenvals[0:self.components_ct]			
		self.eigenvects = self.eigenvects[0:self.components_ct]

		self.eigenvects = self.eigenvects.transpose()									# to column form
		self.eigenvects = self.training_imgs * self.eigenvects	
		self.eigenvects = self.eigenvects / np.linalg.norm(self.eigenvects, axis=0)		# normalize each eigenvector

		self.feats = self.eigenvects.transpose() * self.training_imgs 	# computing the weights
		
		# Should be (400, 400) for ATT database with all training images
		logger.info('images dimensions' + str(np.shape(self.feats)))


	def flattenImages(self, X):
		"""
		Flattens a multi-dimensional data item into a matrix.

		Args:
	     	X: List with multi-dimensional data
	    Returns:
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


	def classify(self, path_to_img):
		"""
		Classify an image to one of the eigenfaces using 1-NN search.

		Args:
			path_to_image: path to the image to query.
		Return:
			id of the predicted face match
		"""
		img = ri.read_single_image(path_to_img)
		flat_img = np.array(img, dtype='float64').flatten()
		flat_img = flat_img - np.array(self.mean_img).flatten()
		flat_img = np.reshape(flat_img, (self.img_dim, 1))

		proj = self.eigenvects.transpose() * flat_img

		diff = self.feats - proj				# finding the min ||feature weights - projected||
		norms = np.linalg.norm(diff, axis=0)

		# calculate the id of the 1st nearest neighbor pic
		closest_face_id = (np.argmin(norms) / self.train_faces_count) + 1
		return closest_face_id


	"""
	Evaluate the model using the 4 test faces left
	from every different face in the AT&T set.
	"""
	def compute(self):
		print 'Computing face matches initiated'

		test_count = self.test_faces_count * self.faces_count

		# Should be 80 for Cambridge dataset
		logger.info('test images count' + str(self.images_ct))

		test_correct = 0
		for i in xrange(self.faces_count):
			for test_id in xrange(1, 11):

				# only examine images not included in the training set
				if test_id not in self.training_ids[i]:
					
					face_id = i + 1
					subj_name = 's' + str(i + 1)
					img_name = str(test_id) + '.pgm'
					path_to_query = os.path.join(self.path_to_faces, subj_name, img_name)

					predicted_id = self.classify(path_to_query)
					result = (predicted_id == face_id)

					if result == True:
						test_correct += 1

		print 'Computing face matches complete'
		self.precision = float(100. * test_correct / test_count)
		print '\nCorrect: ' + str(self.precision) + '%'


