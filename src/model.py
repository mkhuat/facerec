import os
import PIL
import logging
import pprint
import numpy as np
import read_image as ri


logger = logging.getLogger(__name__)

logging.basicConfig(filename='log_' + os.path.basename(__file__),
                            filemode='w',
                            format='%(asctime)s %(name)s %(levelname)s:\t %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

IMG_DIM = 92 * 112


class Model(object):


	def __init__(self, X, ids):
		"""
		Args
			X: the list of all images in our dataset
			ids: id numbers of all images

		Returns
			Weights of the PCA features
		"""

		self.ids = np.asarray(ids)
		self.imgs = self.flattenImages(X)

		# We choose the number of components to be the 
		# number of images
		self.images_ct = self.imgs.shape[1]
		self.components_ct = self.images_ct	# we can set the number of principal
											# components here

		# Should be 400 for ATT database
		logger.info('images count' + str(self.images_ct))

		# Compute the mean of all images over the columns in our matrix	
		self.mean_img = np.sum(self.imgs, axis=1) / float(self.images_ct)

		# Recenter all images
		self.imgs = self.imgs - self.mean_img

		covar = np.matrix(self.imgs.transpose()) * np.matrix(self.imgs)	# linear algebra manipulation for more efficient computation
		covar /= self.images_ct

		# Order the eigenvects descending by their eigenvalue
		self.eigenvals, self.eigenvects = np.linalg.eig(covar)	# eigenvects/values of the covariance matrix
		indexes = np.argsort(-self.eigenvals)				# grab indices that would sort the eigenvals desc.
		self.eigenvals = self.eigenvals[indexes]					# descending order
		self.eigenvects = self.eigenvects[indexes]				# corresponding eigenvects

		# We only use principal components corresponding to 
		# the components_ct largest eigenvalues
		self.eigenvals = self.eigenvals[0:self.components_ct]			
		self.eigenvects = self.eigenvects[0:self.components_ct]

		self.eigenvects = self.eigenvects.transpose()					# change eigenvects from rows to columns
		self.eigenvects = self.imgs * self.eigenvects						# left multiply to get the correct eigenvects
		norms = np.linalg.norm(self.eigenvects, axis=0)		# find the norm of each eigenvector
		self.eigenvects = self.eigenvects / norms						# normalize all eigenvects

		self.feats = self.eigenvects.transpose() * self.imgs 			# computing the weights
		
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
		img = ri.read_single_image(path_to_img)                                        # read as a grayscale image
		img_col = np.array(img, dtype='float64').flatten()                      # flatten the image
		img_col -= self.mean_img                                            # subract the mean column

		img_col = np.reshape(img_col, (IMG_DIM, 1))                             # from row vector to col vector

		S = self.eigenvects.transpose() * img_col                                 # projecting the normalized probe onto the
		                                                                        # Eigenspace, to find out the weights

		diff = self.feats - S                                                       # finding the min ||W_j - S||
		norms = np.linalg.norm(diff, axis=0)

		closest_face_id = np.argmin(norms)                                      # the id [0..240) of the minerror face to the sample
		return (closest_face_id / 10) + 1                   # return the faceid (1..40)


