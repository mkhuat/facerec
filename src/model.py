import numpy as np

# def asColumnMatrix(X):
#     """
#     Creates a column-matrix from multi-dimensional data items.
    
#     Args:
#     	X: List with multi-dimensional data. 
#     	   [images, id_nums] where images is [image, metadata]
#     """
    
#     images = np.array(X[0])						# an array of images of form [image, metadata]
#     if len(X) == 0:
#         return np.array([])

#     total = 1
#     for i in range(0, np.ndim(images)):	# loop through all images
#         total *= images.shape[i]		# multiplying total by 2, in our case

#     result = np.empty([total, 0], dtype=images.dtype)
    
#     for col in X:
#     	# number of rows is inferred, but the number of cols is 1 here
#         col = np.array(col)
#         result = np.append(result, col.reshape(-1,1), axis=1)
    
#     return np.asmatrix(result)

def flattenImage(X):
	"""
	Flattens a multi-dimensional data item into a matrix.

	Args:
     	X: List with multi-dimensional data. 
     	   [images, id_nums] where images is [image, metadata]
	"""
	if len(X) == 0:
		return np.array([])

	images = np.array(X[0])

	total = 1
	for i in range(0, np.ndim(images)):	# loop through all images
		total *= images.shape[i]		# multiplying total by 2, in our case


	result = np.empty(shape=(total, 0), dtype=images.dtype)

	for col in X:
		result = np.hstack((result, np.asarray(col).reshape(-1, 1)))
	return result



  
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