import numpy as np
from scipy.spatial import distance_matrix

def von_neumann_entropy(ro):
	"""
	Computes the Von Neumann Entropy of a matrix. Also known as the Spectral Entropy. 
	In short -- the entropy of the eigenvalues of this matrix.
	INPUT: ro, a n x n matrix. Assumed symmetric and normalized.
	OUTPUT: scalar, the Von Neumann Entropy (what did you expect?)
	"""
	# 1. Eigendecompose ro.
	# TODO: This will be the most time intensive step. Can we improve on numpy?
	evals = np.linalg.eigvalsh(ro)
	# 1.5 TODO: Normalize evals?
	evals = evals / np.sum(evals)
	# 2. Compute entropy of evals
	entropy = -np.sum(evals*np.log(evals))
	return entropy
	
def compute_affinity_matrix(X, kernel_type="gaussian",sigma=0.7, k=3):
	"""
	Given an n x d array of d-dimensional points, creates an affinity matrix (for use with `von_neumann_entropy`)
	"""
	# 1. Get distance matrix (n x n)
	D = distance_matrix(X,X)
	# 2. Turn into affinity matrix, by applying (gaussian) kernel
	if kernel_type == "gaussian":
		s = -np.ones((len(D),len(D)))*(sigma**2)
		W = np.exp(np.divide(np.square(D),s))
	elif kernel_type == "adaptive": # TODO: This can really be improved.
		# a tedious, but accurate implementation
		sigmak_row = np.ones(D.shape)
		for i, row in enumerate(D):
			# go through each row in D. Sort the row, and choose the k^th nearest neighbor.
			# multiply by row i
			sigmak_row[i] = sigmak_row[i] * np.sort(row)[k]
		sigmak_col = np.ones(D.shape)
		for i, col in enumerate(D):
			# go through each row in D. Sort the row, and choose the k^th nearest neighbor.
			# encode this value in the i^th column
			sigmak_col[:,i] = sigmak_col[:,i] * np.sort(col)[k]
		print(sigmak_row)
		W_row = np.divide(D**2,sigmak_row**2)
		W_col = np.divide(D**2,sigmak_col**2)
		W = 0.5*(np.exp(-W_row)+np.exp(-W_col))
	else:
		return "kernel type must be gaussian or adaptive"
	# 3. Normalize for entropy?
	
	# 4. Return
	return W

def von_neumann_entropy_from_matrix(X):
	# Convenience function combining affinity matrix computation and the entropic calculation
	# 1. Get affinity matrix
	W = compute_affinity_matrix(X, "gaussian")	
	# 2. Get entropy
	H = von_neumann_entropy(W)
	return H
	
def disjoint_union(X,Y):
	"""
	Returns the disjoint union of the two sets, treating each row as an element.
	TODO: There is very likely a faster way to do this than looping.
	"""
	OUT = np.empty((X.shape[0]*Y.shape[0],X.shape[1]+Y.shape[1]))
	for i, x in enumerate(X):
		for j, y in enumerate(Y):
			OUT[i*X.shape[0] + j] = np.concatenate([x, y])
	return OUT
	

def spectral_mutual_information(X,Y):
	"""
	Computes spectral mutual information between two matrices, according to
	$$ I(X,Y) = H(X) + H(Y) - H(X,Y) $$
	where H is the Von Neumann entropy of the affinity matrix derived from X.
	"""
	HX = von_neumann_entropy_from_matrix(X)
	HY = von_neumann_entropy_from_matrix(Y)
	# for the joint entropy, we perform the disjoint union of X and Y and get the joint distribution.
	XY = disjoint_union(X, Y)
	HXY = von_neumann_entropy_from_matrix(XY)
	print("HX",HX,"+ HY ",HY,"- HXY ",HXY)
	I = HXY - HX - HY
	return I