"""
Defines methods to calculate the spectral mutual information between graphs.
The function `spectral_mutual_information` takes two matrices (of the same dimension, but with possibly different numbers of rows) and returns the mutual information between the evals of the graph laplacian.
"""

import numpy as np
from scipy.spatial import distance_matrix
from scipy import stats # used to calculate the entropy between the evals
import graphtools
from sklearn.neighbors import kneighbors_graph

def von_neumann_entropy(rho):
	"""
	Computes the Von Neumann Entropy of a matrix. Also known as the Spectral Entropy. 
	In short -- the entropy of the eigenvalues of this matrix.
	INPUT: ro, a n x n matrix. Assumed symmetric and normalized.
	OUTPUT: scalar, the Von Neumann Entropy (what did you expect?)
	"""
	# 1. Eigendecompose ro.
	# TODO: This will be the most time intensive step. Can we improve on numpy?
	evals = np.linalg.eigvalsh(rho)
	# Due to precision errors, the eigendecomposition sometimes gives very small *negative* evals. We'll just take the abs to turn them into (very small) positive ones.
	evals = np.abs(evals)
	# 1.5 TODO: Normalize evals?
	evals = evals / np.sum(evals) # skipping normalization allows the H(X,Y) graph to give bigger entropies.
	print(f"evals {evals}")
	# 2. Compute entropy of evals
	# entropy =  stats.entropy(evals) # scipy has automatic normalization that we don't want, so this is a home-cooked function
	evals_logged = evals.copy()
	evals_logged[evals_logged==0] = 1 # log(0) throws errors, so replace those with something else.
	evals_logged = -np.log(evals_logged)
	entropy = np.multiply(evals,evals_logged)
	#print(f" Entropy: - evals * log evals {entropy}")
	entropy = np.sum(entropy)
	#print(entropy)
	return entropy
	
	
def von_neumann_entropy_from_data(X,double=False):
	# Convenience function combining computation of rho, and the entropic calculation
	# 1. Get normalized laplacian, aka "quantum density matrix", from @passerini2012
	# 1.5 First, use graphtools to build a graph from the data.
	knn=5
	if double:
		print("using double neighbors")
		knn= knn*2
	A = kneighbors_graph(X,knn)
	A = A.toarray() # this has some defaults for automatically selecting the best type of graph. We'll trust them for now.
	# kernel_matrix = G.K # the adjacency matrix with ones on the diagonal
	# A = kernel_matrix - np.eye(kernel_matrix.shape[0]) # get rid of those ones
	# Swap out this function for another to test different methods
	print("A matrix from sklearn",A)
	rho = normalized_laplacian_from_graph(A)
	print("Rho matrix",rho)
	# 2. Compute entropy of evals
	H = von_neumann_entropy(rho)
	return H

def von_neumann_entropy_from_graph(A):
	# Convenience function combining computation of rho, and the entropic calculation
	# 1. Get normalized laplacian, aka "quantum density matrix", from @passerini2012
	rho = normalized_laplacian_from_graph(A)
	#print("Rho matrix",rho)
	# 2. Compute entropy of evals
	H = von_neumann_entropy(rho)
	return H
	
	
def normalized_laplacian_from_graph(A):
	# We compute the "quantum density matrix" rho as the normalized graph laplacian, as suggested by @passerini2012 "The Von Neumann Entropy of Networks"
		# 2. Compute the degree matrix
	degree_sums = np.sum(A,axis=1)
	total_degree_sum = np.sum(degree_sums,axis=None)
	D = np.diag(degree_sums)
	# 3. Compute laplacian
	L = D - A
	# 4. Normalize (in style of @passerini2021) by dividing by the total degree sum
	L = L/total_degree_sum
	print(f"total degree sum {total_degree_sum}")
	return L

	
def spectral_mutual_information(X,Y,is_adjacency_matrix=False):
	"""
	Computes spectral mutual information between two matrices, according to
	$$ I(X,Y) = H(X) + H(Y) - H(X,Y) $$
	where H is the Von Neumann entropy of the affinity matrix derived from X.
	"""
	#print("X",X)
	#print("Y",Y)
	if is_adjacency_matrix:
		# TODO: it would be great to define this directly from a graph; but how do we consider the joint entropy on a graph? It's impossible with just the given information.
		HX = von_neumann_entropy_from_graph(X)
		HY = von_neumann_entropy_from_graph(Y)
		# for the joint entropy, we perform the disjoint union of X and Y and get the joint distribution.
		XY = disjoint_union(X, Y)
		# XY = np.concatenate([X,Y],axis=0)
		#print(XY)
		HXY = von_neumann_entropy_from_graph(XY,double=True)
		print("HX",HX,"+ HY ",HY," - HXY ",HXY)
		I =  HX + HY - HXY
	if not is_adjacency_matrix:
		HX = von_neumann_entropy_from_data(X)
		HY = von_neumann_entropy_from_data(Y)
		# for the joint entropy, we perform the disjoint union of X and Y and get the joint distribution.
		# XY = disjoint_union(X, Y)
		XY = np.concatenate([X,Y],axis=0)
		#print(XY)
		HXY = von_neumann_entropy_from_data(XY,double=True)
		print("HX",HX,"+ HY ",HY," - HXY ",HXY)
		I =  HX + HY - HXY
	return I

def spectral_mutual_information_via_affinity(X,Y):
	"""
	Computes spectral mutual information between two matrices, according to
	$$ I(X,Y) = H(X) + H(Y) - H(X,Y) $$
	where H is the Von Neumann entropy of the affinity matrix derived from X.
	"""
	#print("X",X)
	#print("Y",Y)
	HX = von_neumann_entropy_from_data_via_affinity(X)
	HY = von_neumann_entropy_from_data_via_affinity(Y)
	# for the joint entropy, we perform the disjoint union of X and Y and get the joint distribution.
	# XY = disjoint_union(X, Y)
	XY = np.concatenate([X,Y],axis=0)
	#print(XY)
	HXY = von_neumann_entropy_from_data_via_affinity(XY)
	print("HX",HX,"+ HY ",HY," - HXY ",HXY)
	I =  HX + HY - HXY
	return I


def von_neumann_entropy_from_data_via_affinity(X):
	# Convenience function combining computation of rho, and the entropic calculation
	# We compute the "quantum density matrix" rho as the normalized graph laplacian, as suggested by @passerini2012 "The Von Neumann Entropy of Networks"
	# 1. Get affinity matrix
	W = compute_affinity_matrix(X, "gaussian")	
	# 1.5 Remove self loops
	A = W - np.eye(W.shape[0])
	# 2. Binarize into adjacency matrix. TODO: Loses information. Reconsider. Or build a KNN graph.
	# compute the average connection strength; use as threshold.
	#print(f"W {W}")
	#avg_connection = np.mean(W)
	# A = (W >= np.ones(W.shape)*avg_connection).astype(int)
	print(f"adjacency {A}")
	# 3. Get laplacian
	rho = normalized_laplacian_from_graph(A)
	# 2. Get entropy
	H = von_neumann_entropy(rho)
	return H
	
def disjoint_union(X,Y):
	"""
	! Not used. 
	Returns the disjoint union of the two sets, treating each row as an element.
	TODO: There's definitely a faster way to do this than looping.
	"""
	OUT = np.zeros((X.shape[0]*Y.shape[0],X.shape[1]+Y.shape[1]))
	for i, x in enumerate(X):
		for j, y in enumerate(Y):
			OUT[i*Y.shape[0] + j] = np.concatenate([x, y])
	return OUT

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
	# 4. Return
	return W