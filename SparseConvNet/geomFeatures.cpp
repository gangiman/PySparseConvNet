/*
 * geomFeatures.cpp
 *
 *  Created on: Mar 16, 2016
 *      Author: dmitry.yarotsky
 */

#include <armadillo>
#include "SparseGrid.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <set>
#include <ctime>
#include "geomFeatures.h"


int nFeaturesPerVoxel(enum FeatureKind featureKind){
    int nFeatures;
    if (featureKind == Bool) nFeatures = 1;
	else if (featureKind == ScalarArea) nFeatures = 1;
	else if (featureKind == AreaNormal)	nFeatures = 3;
	else if (featureKind == Quadform) nFeatures = 6;
	else if (featureKind == Eigenvalues) nFeatures = 3;
    return nFeatures;
}

int nFeaturesPerVoxel_set(std::set<enum FeatureKind> featureSet){
    int nFeatures = 0;
    for (const auto& featureKind: featureSet) nFeatures += nFeaturesPerVoxel(featureKind);
    return nFeatures;
}


struct Node { // voxel on some scale
	int level; // number of binary splits from root node
	double LB[3]; // lower bounds
	double UB[3]; // upper bounds
	long Npoly; // number of mesh faces intersecting the voxel
	double *polygons; // intersections with faces; for each polygon 30 doubles are reserved;
	// by construction, polygons cannot contain more than 9 vertices
	int *polySizes; // number of vertices in polygons
	long *faceInd; // map from polygon to index of original face
	struct Node *children; // two children
	struct Node *parent;
	int splitDim; // split dimension producing children (0, 1 or 2)
};


void destroy_tree(struct Node node)
{
	if(node.children != NULL){
		int n;
		for (n=0; n<2; n++) destroy_tree(node.children[n]);
	}
	free(node.children);
	free(node.polygons);
	free(node.polySizes);
	free(node.faceInd);
}


struct Node initRootNode(long Nfaces, double *vertA, long *faceA, double bound){
	struct Node rootNode;
	int n, k, d;
	for (n=0; n<3; n++){
		rootNode.LB[n] = -bound;
		rootNode.UB[n] = bound;
	}
	rootNode.polygons = (double*) malloc(Nfaces*10*3*sizeof(double));
	rootNode.polySizes = (int*) malloc(Nfaces*sizeof(int));
	rootNode.faceInd = (long*) malloc(Nfaces*sizeof(long));
	for (n=0; n<Nfaces; n++){
		rootNode.polySizes[n] = 3;
		rootNode.faceInd[n] = n;
		for (k=0; k<3; k++){
			for (d=0; d<3; d++){
				rootNode.polygons[n*10*3+k*3+d] = vertA[3*faceA[n*3+k]+d];
			}
		}
	}
	rootNode.level = 0;
	rootNode.children = NULL;
	assert(rootNode.children == NULL);
	rootNode.Npoly = Nfaces;

	return rootNode;
}

struct Node initRootNode(long Nfaces, arma::mat points, std::vector<std::vector<int>> surfaces, double bound){
	struct Node rootNode;
	int n, k, d;
	for (n=0; n<3; n++){
		rootNode.LB[n] = -bound;
		rootNode.UB[n] = bound;
	}
	rootNode.polygons = (double*) malloc(Nfaces*10*3*sizeof(double));
	rootNode.polySizes = (int*) malloc(Nfaces*sizeof(int));
	rootNode.faceInd = (long*) malloc(Nfaces*sizeof(long));
	for (n=0; n<Nfaces; n++){
		rootNode.polySizes[n] = 3;
		rootNode.faceInd[n] = n;
		for (k=0; k<3; k++){
			for (d=0; d<3; d++){
				rootNode.polygons[n*10*3+k*3+d] = points(surfaces[n][k],d);
			}
		}
	}
	rootNode.level = 0;
	rootNode.children = NULL;
	assert(rootNode.children == NULL);
	rootNode.Npoly = Nfaces;

	return rootNode;
}


struct Node* splitNode(struct Node *parentPtr, int dim){
	parentPtr->children = (struct Node*) malloc(2*sizeof(struct Node));
	struct Node *children = parentPtr->children;

	int n, k, k1, d;
	double a;
	double eps = 1e-12;
	parentPtr->splitDim = dim;

	for (n=0; n<2; n++){
		for (d=0; d<3; d++){
			children[n].LB[d] = parentPtr->LB[d];
			children[n].UB[d] = parentPtr->UB[d];
		}
	}
	double splitVal = 0.5*(parentPtr->LB[dim]+parentPtr->UB[dim]);
	children[0].UB[dim] = splitVal;
	children[1].LB[dim] = splitVal;

	for (n=0; n<2; n++){
		children[n].level = parentPtr->level+1;
		children[n].Npoly = 0;
		children[n].children = NULL;
		children[n].polygons = (double*) malloc((parentPtr->Npoly)*10*3*sizeof(double));
		children[n].polySizes = (int*) malloc((parentPtr->Npoly)*sizeof(int));
		children[n].faceInd = (long*) malloc((parentPtr->Npoly)*sizeof(long));
		children[n].parent = parentPtr;
	}

	for (n=0; n < parentPtr->Npoly; n++){ // construct polygons for children
		int Nneg = 0, Npos = 0;
		for (k=0; k<parentPtr->polySizes[n]; k++){
			Npos += (parentPtr->polygons[n*10*3+k*3+dim] >= splitVal+eps) ? 1 : 0;
			Nneg += (parentPtr->polygons[n*10*3+k*3+dim] <= splitVal-eps) ? 1 : 0;
		}
		if (Npos == 0){ // second child is empty
			for (k=0; k<parentPtr->polySizes[n]; k++){
				for (d=0; d<3; d++){
					children[0].polygons[(children[0].Npoly)*10*3+k*3+d] \
					= parentPtr->polygons[n*10*3+k*3+d];
				}
			}
			children[0].polySizes[children[0].Npoly] = parentPtr->polySizes[n];
			children[0].faceInd[children[0].Npoly] = parentPtr->faceInd[n];
			children[0].Npoly += 1;
		} else if (Nneg == 0){ // first child is empty
			for (k=0; k<parentPtr->polySizes[n]; k++){
				for (d=0; d<3; d++){
					children[1].polygons[(children[1].Npoly)*10*3+k*3+d] \
					= parentPtr->polygons[n*10*3+k*3+d];
				}
			}
			children[1].polySizes[children[1].Npoly] = parentPtr->polySizes[n];
			children[1].faceInd[children[1].Npoly] = parentPtr->faceInd[n];
			children[1].Npoly += 1;
		} else { // both children are nonempty
			children[0].faceInd[children[0].Npoly] = parentPtr->faceInd[n];
			children[1].faceInd[children[1].Npoly] = parentPtr->faceInd[n];

			children[0].polySizes[children[0].Npoly] = 0;
			children[1].polySizes[children[1].Npoly] = 0;
			for (k=0; k<parentPtr->polySizes[n]; k++){
				k1 = (k+1)%(parentPtr->polySizes[n]);
				// intersection with splitting plane at an internal point of the edge; add new vertex at intersection
				if (parentPtr->polygons[n*10*3+k*3+dim] <= splitVal-eps &&
						parentPtr->polygons[n*10*3+k1*3+dim] >= splitVal+eps){
					for (d=0; d<3; d++){
						children[0].polygons[(children[0].Npoly)*10*3+ \
											 children[0].polySizes[children[0].Npoly]*3+d] \
											 = parentPtr->polygons[n*10*3+k*3+d];
					}
					children[0].polySizes[children[0].Npoly] += 1;

					a = (splitVal-parentPtr->polygons[n*10*3+k1*3+dim])/ \
							(parentPtr->polygons[n*10*3+k*3+dim]-parentPtr->polygons[n*10*3+k1*3+dim]);
					for (d=0; d<3; d++){
						children[0].polygons[(children[0].Npoly)*10*3+ \
											 children[0].polySizes[children[0].Npoly]*3+d] \
											 = a*(parentPtr->polygons[n*10*3+k*3+d]) + \
											 (1-a)*(parentPtr->polygons[n*10*3+k1*3+d]);
					}
					children[0].polySizes[children[0].Npoly] += 1;

					for (d=0; d<3; d++){
						children[1].polygons[(children[1].Npoly)*10*3+ \
											 children[1].polySizes[children[1].Npoly]*3+d] \
											 = a*(parentPtr->polygons[n*10*3+k*3+d]) + \
											 (1-a)*(parentPtr->polygons[n*10*3+k1*3+d]);
					}
					children[1].polySizes[children[1].Npoly] += 1;
				} else if (parentPtr->polygons[n*10*3+k*3+dim] >= splitVal+eps &&
						parentPtr->polygons[n*10*3+k1*3+dim] <= splitVal-eps){
					for (d=0; d<3; d++){
						children[1].polygons[(children[1].Npoly)*10*3+ \
											 children[1].polySizes[children[1].Npoly]*3+d] \
											 = parentPtr->polygons[n*10*3+k*3+d];
					}
					children[1].polySizes[children[1].Npoly] += 1;
					a = (splitVal-parentPtr->polygons[n*10*3+k1*3+dim])/ \
							(parentPtr->polygons[n*10*3+k*3+dim]-parentPtr->polygons[n*10*3+k1*3+dim]);
					for (d=0; d<3; d++){
						children[1].polygons[(children[1].Npoly)*10*3+ \
											 children[1].polySizes[children[1].Npoly]*3+d] \
											 = a*(parentPtr->polygons[n*10*3+k*3+d]) + \
											 (1-a)*(parentPtr->polygons[n*10*3+k1*3+d]);
					}
					children[1].polySizes[children[1].Npoly] += 1;
					for (d=0; d<3; d++){
						children[0].polygons[(children[0].Npoly)*10*3+ \
											 children[0].polySizes[children[0].Npoly]*3+d] \
											 = a*(parentPtr->polygons[n*10*3+k*3+d]) + \
											 (1-a)*(parentPtr->polygons[n*10*3+k1*3+d]);
					}
					children[0].polySizes[children[0].Npoly] += 1;
				} else{ // intersection with splitting plane may be only at the edge end point
					if (parentPtr->polygons[n*10*3+k*3+dim] > splitVal-eps) {
						for (d=0; d<3; d++){
							children[1].polygons[(children[1].Npoly)*10*3+ \
												 children[1].polySizes[children[1].Npoly]*3+d] \
												 = parentPtr->polygons[n*10*3+k*3+d];
						}
						children[1].polySizes[children[1].Npoly] += 1;
					}
					if (parentPtr->polygons[n*10*3+k*3+dim] < splitVal+eps) {
						for (d=0; d<3; d++){
							children[0].polygons[(children[0].Npoly)*10*3+ \
												 children[0].polySizes[children[0].Npoly]*3+d] \
												 = parentPtr->polygons[n*10*3+k*3+d];
						}
						children[0].polySizes[children[0].Npoly] += 1;
					}
				}
			}
			children[0].Npoly += 1;
			children[1].Npoly += 1;
		}

	}
	return children;
}


void splitIter(struct Node *rootPtr, int depth){ // split iteratively up to given binary depth
	if (rootPtr->Npoly > 0){
		splitNode(rootPtr, rootPtr->level%3);
		if (rootPtr->level < depth-1){
			int s;
			for (s=0; s<2; s++){
				splitIter(rootPtr->children+s, depth);
			}
		}
	}
}


double computeScalarArea(struct Node *nodePtr){ // total area of all intersections with faces
	int n, k, d;
	double a[3], b[3];
	arma::vec areaNormalVector(3);
	double doubleArea = 0;
	for (n=0; n < nodePtr->Npoly; n++){
		for (d=0; d<3; d++) areaNormalVector[d] = 0;
		for (k=2; k < nodePtr->polySizes[n]; k++){
			for (d=0; d<3; d++){
				a[d] = nodePtr->polygons[n*10*3+k*3+d]-nodePtr->polygons[n*10*3+d];
				b[d] = nodePtr->polygons[n*10*3+(k-1)*3+d]-nodePtr->polygons[n*10*3+d];
			}
			areaNormalVector[0] += a[1]*b[2]-b[1]*a[2];
			areaNormalVector[1] += a[2]*b[0]-b[2]*a[0];
			areaNormalVector[2] += a[0]*b[1]-b[0]*a[1];
		}
		doubleArea += sqrt(areaNormalVector[0]*areaNormalVector[0]+
				areaNormalVector[1]*areaNormalVector[1]+
				areaNormalVector[2]*areaNormalVector[2]);
	}
	return doubleArea/2;
}


arma::vec computeAreaNormal(struct Node *nodePtr){ // sum of area vectors of all intersections with faces
	int n, k, d;
	double a[3], b[3];
	arma::vec areaNormalVector(3);
	for (n=0; n<3; n++) areaNormalVector[n] = 0;

	for (n=0; n < nodePtr->Npoly; n++){
		for (k=2; k < nodePtr->polySizes[n]; k++){
			for (d=0; d<3; d++){
				a[d] = nodePtr->polygons[n*10*3+k*3+d]-nodePtr->polygons[n*10*3+d];
				b[d] = nodePtr->polygons[n*10*3+(k-1)*3+d]-nodePtr->polygons[n*10*3+d];
			}
			areaNormalVector[0] += a[1]*b[2]-b[1]*a[2];
			areaNormalVector[1] += a[2]*b[0]-b[2]*a[0];
			areaNormalVector[2] += a[0]*b[1]-b[0]*a[1];
		}
	}
	areaNormalVector[0] /= 2;
	areaNormalVector[1] /= 2;
	areaNormalVector[2] /= 2;

	return areaNormalVector;
}


arma::vec computeQuadform(struct Node *nodePtr){ // sum of quadratic forms for area vectors of all intersections with faces
	int n, k, d;
	double a[3], b[3];
	arma::vec quadformVector(6);
	for (n=0; n<6; n++) quadformVector[n] = 0;
	double doubleArea, c;

	arma::vec areaNormalVector(3);
	for (n=0; n < nodePtr->Npoly; n++){
		for (d=0; d<3; d++) areaNormalVector[d] = 0;
		for (k=2; k < nodePtr->polySizes[n]; k++){
			for (d=0; d<3; d++){
				a[d] = nodePtr->polygons[n*10*3+k*3+d]-nodePtr->polygons[n*10*3+d];
				b[d] = nodePtr->polygons[n*10*3+(k-1)*3+d]-nodePtr->polygons[n*10*3+d];
			}
			areaNormalVector[0] += a[1]*b[2]-b[1]*a[2];
			areaNormalVector[1] += a[2] * b[0] - b[2] * a[0];
			areaNormalVector[2] += a[0] * b[1] - b[0] * a[1];
		}
		doubleArea = sqrt(
				areaNormalVector[0] * areaNormalVector[0]
						+ areaNormalVector[1] * areaNormalVector[1]
						+ areaNormalVector[2] * areaNormalVector[2]);
		if (doubleArea > 0) {
			c = 0.5 / doubleArea;
			quadformVector[0] += areaNormalVector[0] * areaNormalVector[0] * c;
			quadformVector[1] += areaNormalVector[1] * areaNormalVector[1] * c;
			quadformVector[2] += areaNormalVector[2] * areaNormalVector[2] * c;
			quadformVector[3] += areaNormalVector[0] * areaNormalVector[1] * c;
			quadformVector[4] += areaNormalVector[0] * areaNormalVector[2] * c;
			quadformVector[5] += areaNormalVector[1] * areaNormalVector[2] * c;
		}
	}
	return quadformVector;
}

arma::vec computeEigenvalues(struct Node *nodePtr){ // eigenvalues of the quadform
	arma::vec quadformVector(6);
	quadformVector = computeQuadform(nodePtr);
	arma::mat Q(3,3);
	Q << quadformVector[0] << quadformVector[3] << quadformVector[4] << arma::endr
	  << quadformVector[3] << quadformVector[1] << quadformVector[5] << arma::endr
	  << quadformVector[4] << quadformVector[5] << quadformVector[2] << arma::endr;

	arma::vec eigenvalueVector(3);
	arma::eig_sym(eigenvalueVector, Q);

	return eigenvalueVector;
}


void fillFeatureData_set(struct Node *rootPtr,
		             int maxlevel,
					 long *m,
					 int spatialSize,
		             SparseGrid &grid,
					 std::vector<float> &features,
					 std::set<enum FeatureKind> featureSet){
	// given binary tree, iteratively fill sparse grid data and feature data
	// TODO: optimize: Quadform is computed twice if present along with Eigenvalues
	if (rootPtr->level == maxlevel && rootPtr->Npoly > 0){
		int d, x[3], n;
		for (d=0; d<3; d++){
			x[d] = floor(0.5*(rootPtr->LB[d]+rootPtr->UB[d]))+spatialSize/2;
			assert(x[d] >= 0);
			assert(x[d] < spatialSize);
		}

		n = x[0]*spatialSize*spatialSize+x[1]*spatialSize+x[2];
		grid.mp[n] = *m;

		for (const auto& featureKind: featureSet) {
			if (featureKind == Bool){
				features.push_back(1);
			} else if (featureKind == ScalarArea){
				double area = computeScalarArea(rootPtr);
				features.push_back((float) area);
			} else if (featureKind == AreaNormal){
				arma::vec areaNormal = computeAreaNormal(rootPtr);
				for (n=0; n<3; n++) features.push_back(areaNormal[n]);
			} else if (featureKind == Quadform){
				arma::vec quadform = computeQuadform(rootPtr);
				for (n=0; n<6; n++) features.push_back(quadform[n]);
			}else if (featureKind == Eigenvalues){
				arma::vec eigenvalues = computeEigenvalues(rootPtr);
				for (n=0; n<3; n++) features.push_back(eigenvalues[n]);
			}
		}
		*m += 1;
	} else if (rootPtr->children != NULL){
		int s;
		for (s=0; s<2; s++){
			fillFeatureData_set(rootPtr->children+s, maxlevel, m, spatialSize, grid, features, featureSet);
		}
	}
}


void get_features_set(arma::mat points,
		          std::vector<std::vector<int>> surfaces,
				  SparseGrid &grid,
				  std::vector<float> &features,
				  int &nSpatialSites,
				  int spatialSize,
				  std::set<enum FeatureKind> featureSet){
	/*
	 Input data:
	    points, surfaces: Provided shape data (.off format)
	    spatialSize: Linear size of the cube containing the shape.
	                 The cube is centered at the origin, so each side is [-spatialSize/2, spatialSize/2].
	                 It is assumed that spatialSize is even.
	                 The cube is divided into (spatialSize x spatialSize x spatialSize) voxels of unit volume (1 x 1 x 1).
	                 It is assumed that the shape has already been appropriately normalized to fit into the cube.
	     featureSet: The set of features to be evaluated.
	                 Each voxel is assigned a feature vector of a fixed length equal to the sum of nFeaturesPerVoxel
	                      over all featureKinds in featureSet.
	 Output data:
	    grid: Dictionary of nonempty cells.
	          For the voxel with coordinates x,y,z ranging between 0 and spatialSize-1, grid maps x*spatialSize*spatialSize+y*spatialSize+z
	          into the number enumerating that voxel.
	    nSpatialSites: Number of nonempty voxels (equals the size of the dictionary in grid).
	    features: Feature vector of size nSpatialSites*nFeaturesPerVoxel (features for voxel 0, features for voxel 1, ...).

	 */

	assert(points.n_cols == 3);

        for (size_t i = 0; i < nFeaturesPerVoxel_set(featureSet); ++i) {
	  features.push_back(0); // Background feature
	}

        unsigned int row;
	unsigned int col;

	// check shape fits in given cube
	double maxAbsVal = 0;
	for (col=0; col<3; col++){
		for (row=0; row<points.n_rows; row++){
			maxAbsVal = fmax(fabs(points(row, col)), maxAbsVal);
		}
	}
	assert(2*(spatialSize/2) == spatialSize);
	assert(maxAbsVal <= spatialSize/2);

	int depth = (int)ceil(log2((double)spatialSize));

	// lowest integer bound for spatialSize/2 of the dyadic form 2**n
	int halfSideDyadic = (int) pow(2, depth-1);
	double bound = (double) halfSideDyadic;

	struct Node rootNode = initRootNode((long) surfaces.size(), points, surfaces, bound);
	splitIter(&rootNode, 3*depth); // depth in each of 3 dimensions

	long m = 0;
	fillFeatureData_set(&rootNode, 3*depth, &m, spatialSize, grid, features, featureSet);
	nSpatialSites = features.size()/nFeaturesPerVoxel_set(featureSet);

	destroy_tree(rootNode);
}

void get_features(arma::mat points,
		          std::vector<std::vector<int>> surfaces,
				  SparseGrid &grid,
				  std::vector<float> &features,
				  int &nSpatialSites,
				  int spatialSize,
				  enum FeatureKind featureKind){
	/*
	 Input data:
	    points, surfaces: Provided shape data (.off format)
	    spatialSize: Linear size of the cube containing the shape.
	                 The cube is centered at the origin, so each side is [-spatialSize/2, spatialSize/2].
	                 It is assumed that spatialSize is even.
	                 The cube is divided into (spatialSize x spatialSize x spatialSize) voxels of unit volume (1 x 1 x 1).
	                 It is assumed that the shape has already been appropriately normalized to fit into the cube.
	    featureKind: The kind of features to be evaluated.
	                 Each voxel is assigned a feature vector of a fixed length nFeaturesPerVoxel depending on the feature kind.

	 Output data:
	    grid: Dictionary of nonempty cells.
	          For the voxel with coordinates x,y,z ranging between 0 and spatialSize-1, grid maps x*spatialSize*spatialSize+y*spatialSize+z
	          into the number enumerating that voxel.
	    nSpatialSites: Number of nonempty voxels (equals the size of the dictionary in grid).
	    features: Feature vector of size nSpatialSites*nFeaturesPerVoxel (features for voxel 0, features for voxel 1, ...).

	 */

	std::set<enum FeatureKind> featureSet = {featureKind};
	get_features_set(points,
			         surfaces,
					 grid,
					 features,
					 nSpatialSites,
					 spatialSize,
					 featureSet);

}
