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

	# If we have no images to process, just return an empty array
	# if len(X) == 0:
	# 	return np.array([])


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


  
# class PCA(AbstractFeature):
#     def __init__(self, num_components=0):
#         AbstractFeature.__init__(self)
#         self._num_components = num_components
        
#     def compute(self,X,y):
#         # build the column matrix
#         XC = asColumnMatrix(X)
#         y = np.asarray(y)

#         # set a valid number of components
#         if self._num_components <= 0 or (self._num_components > XC.shape[1]-1):
#             self._num_components = XC.shape[1]-1

#         # center dataset
#         self._mean = XC.mean(axis=1).reshape(-1,1)
#         XC = XC - self._mean

#         # perform an economy size decomposition (may still allocate too much memory for computation)
#         self._eigenvectors, self._eigenvalues, variances = np.linalg.svd(XC, full_matrices=False)

#         # sort eigenvectors by eigenvalues in descending order
#         idx = np.argsort(-self._eigenvalues)
#         self._eigenvalues, self._eigenvectors = self._eigenvalues[idx], self._eigenvectors[:,idx]

#         # use only num_components
#         self._eigenvectors = self._eigenvectors[0:,0:self._num_components].copy()
#         self._eigenvalues = self._eigenvalues[0:self._num_components].copy()

#         # finally turn singular values into eigenvalues 
#         self._eigenvalues = np.power(self._eigenvalues,2) / XC.shape[1]

#         # get the features from the given data
#         features = []
#         for x in X:
#             xp = self.project(x.reshape(-1,1))
#             features.append(xp)
#         return features
    
#     def extract(self,X):
#         X = np.asarray(X).reshape(-1,1)
#         return self.project(X)
        
#     def project(self, X):
#         X = X - self._mean
#         return np.dot(self._eigenvectors.T, X)

#     def reconstruct(self, X):
#         X = np.dot(self._eigenvectors, X)
#         return X + self._mean

#     @property
#     def num_components(self):
#         return self._num_components

#     @property
#     def eigenvalues(self):
#         return self._eigenvalues
        
#     @property
#     def eigenvectors(self):
#         return self._eigenvectors

#     @property
#     def mean(self):
#         return self._mean
        
#     def __repr__(self):
#         return "PCA (num_components=%d)" % (self._num_components)