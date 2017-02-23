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


def flattenImage(X):
	"""
	Flattens a multi-dimensional data item into a matrix.

	Args:
     	X: List with multi-dimensional data. 
     	   [images, id_nums] where images is [image, metadata]
	"""

	# If we have no image to process, just return an empty array
	if len(X) == 0:
		return np.array([])

	x0 = np.asarray(X[0])	# Assume that image is standardized

	# Initialize a matrix that represents the flattened image
	flat_img = np.empty((x0.size, 0), dtype=x0.dtype)

	# Grab each column in the multidimensional image 
	# and add to our horizontal matrix
	for col in X:
		flat_img = np.append(flat_img, np.asarray(col).reshape(-1,1), axis=1)
	
	return np.asmatrix(flat_img)