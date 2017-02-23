import os
import PIL
import logging
import numpy as np
from builtins import range

logger = logging.getLogger(__name__)

logging.basicConfig(filename='log_' + os.path.basename(__file__),
                            filemode='w',
                            format='%(asctime)s %(name)s %(levelname)s:\t %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)


def flattenImages(X):
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


def pca(X, ids):
	pass


