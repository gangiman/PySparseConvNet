/*
 * geomFeatures.cpp
 *
 *  Created on: Mar 16, 2016
 *      Author: dmitry.yarotsky
 */

#include "geomFeatures.h"
#include <armadillo>
#include "SparseGrid.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <set>
#include <ctime>
#include "types.h"


int nFeaturesPerVoxel(enum FeatureKind featureKind){
  int nFeatures;
  if (featureKind == Bool) nFeatures = 1;
  else if (featureKind == ScalarArea) nFeatures = 1;
  else if (featureKind == AreaNormal)	nFeatures = 3;
  else if (featureKind == Quadform) nFeatures = 6;
  else if (featureKind == Eigenvalues) nFeatures = 3;
  else if (featureKind == QFoverSA) nFeatures = 6;
  else if (featureKind == EVoverSA) nFeatures = 3;
  else if (featureKind == AngularDefect) nFeatures = 1;
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
  int Nverts; // number of original vertices found in the voxel
  long *vertInd; // map from polygon to index of original vertices
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
  free(node.vertInd);
}


struct Node initRootNode(long Nverts, long Nfaces, arma::mat const &points, std::vector<std::vector<int>> const &surfaces, double bound){
  struct Node rootNode;
  long n, k, d;
  for (n=0; n<3; n++){
    rootNode.LB[n] = -bound;
    rootNode.UB[n] = bound;
  }
  rootNode.polygons = (double*) malloc(Nfaces*10*3*sizeof(double));
  rootNode.polySizes = (int*) malloc(Nfaces*sizeof(int));
  rootNode.faceInd = (long*) malloc(Nfaces*sizeof(long));
  rootNode.vertInd = (long*) malloc(Nverts*sizeof(long));
  for (n=0; n<Nverts; n++) rootNode.vertInd[n] = n;
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
  rootNode.Nverts = Nverts;

  return rootNode;
}


struct Node* splitNode(struct Node *parentPtr, int dim, arma::mat const &points){
  parentPtr->children = (struct Node*) malloc(2*sizeof(struct Node));
  struct Node *children = parentPtr->children;

  long n, k, k1, d;
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
    children[n].vertInd = (long*) malloc((parentPtr->Nverts)*sizeof(long));
    children[n].parent = parentPtr;
    children[n].Nverts = 0;
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
  for (n=0; n < parentPtr->Nverts; n++){ // assign parent's vertices to children
    if (points(parentPtr->vertInd[n], dim) > splitVal+eps){
      children[1].vertInd[children[1].Nverts] = parentPtr->vertInd[n];
      children[1].Nverts += 1;
    } else if (points(parentPtr->vertInd[n], dim) < splitVal-eps){
      children[0].vertInd[children[0].Nverts] = parentPtr->vertInd[n];
      children[0].Nverts += 1;
    } else {
      assert(children[0].Npoly+children[1].Npoly > 0);
      if (children[0].Npoly > 0){
        children[0].vertInd[children[0].Nverts] = parentPtr->vertInd[n];
        children[0].Nverts += 1;
      } else {
        children[1].vertInd[children[1].Nverts] = parentPtr->vertInd[n];
        children[1].Nverts += 1;
      }
    }
  }

  // check consistency of the split
  assert(children[0].Nverts+children[1].Nverts == parentPtr->Nverts);

  int maxInd = 0, maxIndChild[2];
  for (k=0; k<parentPtr->Nverts; k++) maxInd = (maxInd > parentPtr->vertInd[k]) ? maxInd : parentPtr->vertInd[k];
  assert(maxInd < (long) points.n_rows);
  for (n=0; n<2; n++){
    //assert((children[n].Npoly > 0) || (children[n].Nverts == 0));
    maxIndChild[n] = 0;
    for (k=0; k<children[n].Nverts; k++) maxIndChild[n] = (maxIndChild[n] > children[n].vertInd[k]) ? maxInd : children[n].vertInd[k];
    assert(maxIndChild[n] <= maxInd);
  }
  assert((double) fmax(maxIndChild[0], maxIndChild[1]) == (double) maxInd);

  return children;
}


void splitIter(struct Node *rootPtr, int depth, arma::mat const &points){ // split iteratively up to given binary depth
  if ((rootPtr->Npoly > 0) || (rootPtr->Nverts > 0)){
    splitNode(rootPtr, rootPtr->level%3, points);
    if (rootPtr->level < depth-1){
      int s;
      for (s=0; s<2; s++){
        splitIter(rootPtr->children+s, depth, points);
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
                         std::set<enum FeatureKind> featureSet,
                         std::vector<double> const &defects){
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
      } else if (featureKind == QFoverSA){
        double area = computeScalarArea(rootPtr);
        arma::vec quadform = computeQuadform(rootPtr);
        for (n=0; n<6; n++) features.push_back(quadform[n]/(area+1e-10f));
      } else if (featureKind == Eigenvalues){
        arma::vec eigenvalues = computeEigenvalues(rootPtr);
        for (n=0; n<3; n++) features.push_back(eigenvalues[n]);
      } else if (featureKind == EVoverSA){
        double area = computeScalarArea(rootPtr);
        arma::vec eigenvalues = computeEigenvalues(rootPtr);
        for (n=0; n<3; n++) features.push_back(eigenvalues[n]/(area+1e-10f));
      } else if (featureKind == AngularDefect){
        double totalNodeDefect = 0;
        for (n=0; n < rootPtr->Nverts; n++) {
          assert(rootPtr->vertInd[n] < (int) defects.size());
          totalNodeDefect += defects[rootPtr->vertInd[n]];
        }
        features.push_back(totalNodeDefect);
      }
    }
    *m += 1;
  } else if (rootPtr->children != NULL){
    int s;
    for (s=0; s<2; s++){
      fillFeatureData_set(rootPtr->children+s, maxlevel, m, spatialSize, grid, features, featureSet, defects);
    }
  }
}

void computeAngDefects(arma::mat const &points, std::vector<std::vector<int>> const &surfaces, std::vector<double> &defects, int ignoreUnusedVertices){
  unsigned int m;
  defects.clear();
  for (m=0; m<points.n_rows; m++) defects.push_back(0);
  std::vector<int> surface;
  int Nsurf = surfaces.size();
  int n, k, d;
  double v[3][3], norm[3], angle, a;
  for (n=0; n<Nsurf; n++){
    surface = surfaces[n];
    for (k=0; k<3; k++){
      for (d=0; d<3; d++){
        v[k][d] = points(surface[k],d)-points(surface[(k+1)%3],d);
      }
    }
    for (k=0; k<3; k++){
      norm[k] = sqrt(v[k][0]*v[k][0]+v[k][1]*v[k][1]+v[k][2]*v[k][2]);
      assert(norm[k] > 0); // assume no repeated points
    }
    for (k=0; k<3; k++){
      a = -(v[k][0]*v[(k+1)%3][0]+v[k][1]*v[(k+1)%3][1]+v[k][2]*v[(k+1)%3][2])/(norm[k]*norm[(k+1)%3]);
      angle = acos(a);

      // angle = std::min(std::max(angle, 0.0), M_PI);
      // assert((angle >= -1e-5) && (angle <= M_PI + 1e-5));
      defects[surface[(k+1)%3]] += angle;
    }
  }
  for (m=0; m<points.n_rows; m++) defects[m] = 2*M_PI-defects[m];

  if (ignoreUnusedVertices == 1){ // if a vertex does not belong to any surface (as happens in ModelNet), set its defect to 0
    std::vector<int> nSurfaces;
    for (n=0; n<(int)points.n_rows; n++) {
      nSurfaces.push_back(0);
    }
    for (n=0; n<(int)surfaces.size(); n++){
      for (d=0; d<3; d++){
        nSurfaces[surfaces[n][d]] += 1;
      }
    }
    for (n=0; n<(int)points.n_rows; n++) {
      if (nSurfaces[n] == 0) defects[n] = 0;
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

	struct Node rootNode = initRootNode(points.n_rows, (long) surfaces.size(), points, surfaces, bound);
	splitIter(&rootNode, 3*depth, points); // depth in each of 3 dimensions

        std::vector<double> defects;
        if (featureSet.find(AngularDefect) != featureSet.end()){
          computeAngDefects(points, surfaces, defects, 1);
        }


	long m = 0;
        fillFeatureData_set(&rootNode, 3*depth, &m, spatialSize, grid, features, featureSet, defects);
	nSpatialSites = features.size()/nFeaturesPerVoxel_set(featureSet);

	destroy_tree(rootNode);
}


void get_features(arma::mat const &points,
                  std::vector<std::vector<int>> const &surfaces,
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

double getEulerChar(arma::mat const &points, std::vector<std::vector<int>> const &surfaces){
  std::vector<double> defects;
  computeAngDefects(points, surfaces, defects, 1);
  unsigned int n;
  double totalDefect = 0;
  for (n=0; n<points.n_rows; n++) {
    totalDefect += defects[n];
  }
  return totalDefect/(2*M_PI);
}

// void runTests(){
//   int n, k;
//   arma::mat points0;
//   points0 << 0 << 0 << 0 << arma::endr
//           << 0 << 0 << 1 << arma::endr
//           << 0 << 1 << 0 << arma::endr
//           << 1 << 0 << 0 << arma::endr
//           << 0 << 1 << 1 << arma::endr;
//   arma::mat points = points0*5-0.1;

//   std::vector<std::vector<int>> surfaces;
//   std::vector<int> curSurf;
//   curSurf = {0, 1, 2};
//   surfaces.push_back(curSurf);
//   curSurf = {0, 2, 3};
//   surfaces.push_back(curSurf);

//   SparseGrid grid;
//   std::vector<float> features;
//   int nSpatialSites, nSpatialSites1;

//   int spatialSize = 200;
//   enum FeatureKind featureKind = Bool;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites,
//                spatialSize,
//                featureKind);

//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
//   for (n=0; n<nSpatialSites; n++)	assert(features[n] == 1.f);

//   spatialSize = 100;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites1,
//                spatialSize,
//                featureKind);
//   assert(nSpatialSites1 == nSpatialSites);


//   featureKind = ScalarArea;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites1,
//                spatialSize,
//                featureKind);
//   assert(nSpatialSites1 == nSpatialSites);
//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
//   for (n=0; n<nSpatialSites; n++)	assert(features[n] >= 0);
//   for (n=0; n<nSpatialSites; n++)	assert(features[n] <= 4);
//   int someValueEquals1 = 0;
//   for (n=0; n<nSpatialSites; n++){
//     if (fabs(features[n]-1) < 1e-5) someValueEquals1 = 1;
//   }
//   assert(someValueEquals1 == 1);
//   float totalArea = 0;
//   for (n=0; n<nSpatialSites; n++)	totalArea += features[n];
//   assert(fabs(totalArea-25) < 1e-5);


//   featureKind = AreaNormal;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites1,
//                spatialSize,
//                featureKind);
//   assert(nSpatialSites1 == nSpatialSites);
//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
//   for (n=0; n<nSpatialSites; n++)	assert(features[n] >= -2);
//   for (n=0; n<nSpatialSites; n++)	assert(features[n] <= 2);
//   totalArea = 0;
//   for (n=0; n<nSpatialSites*3; n++)	totalArea += fabs(features[n]);
//   assert(fabs(totalArea-25) < 1e-5);


//   featureKind = Quadform;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites1,
//                spatialSize,
//                featureKind);
//   assert(nSpatialSites1 == nSpatialSites);
//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
//   for (n=0; n<nSpatialSites; n++)	assert(features[n] >= -4);
//   for (n=0; n<nSpatialSites; n++)	assert(features[n] <= 4);
//   totalArea = 0;
//   for (n=0; n<nSpatialSites; n++)	{
//     assert(features[6*n] >= 0);
//     assert(features[6*n+1] >= 0);
//     assert(features[6*n+2] >= 0);
//     assert(features[6*n]*features[6*n+1] >= features[6*n+3]*features[6*n+3]);
//     totalArea += fabs(features[6*n]);
//     totalArea += fabs(features[6*n+1]);
//     totalArea += fabs(features[6*n+2]);
//   }
//   assert(fabs(totalArea-25) < 1e-5);


//   featureKind = Eigenvalues;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites1,
//                spatialSize,
//                featureKind);
//   assert(nSpatialSites1 == nSpatialSites);
//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
//   for (n=0; n<nSpatialSites; n++)	assert(features[n] >= 0);
//   for (n=0; n<nSpatialSites; n++)	assert(features[n] <= 2);
//   totalArea = 0;
//   for (n=0; n<nSpatialSites*3; n++)	totalArea += fabs(features[n]);
//   assert(fabs(totalArea-25) < 1e-5);

//   curSurf = {3, 2, 1};
//   surfaces.push_back(curSurf);
//   curSurf = {0, 3, 1};
//   surfaces.push_back(curSurf);


//   featureKind = AreaNormal;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites,
//                spatialSize,
//                featureKind);
//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
//   float normalSum[3];
//   for (n=0; n<3; n++) normalSum[n] = 0;
//   float totalAbsSum = 0;
//   for (n=0; n<nSpatialSites; n++)	{
//     for (k=0; k<3; k++){
//       normalSum[k] += features[3*n+k];
//       totalAbsSum += fabs(features[3*n+k]);
//     }
//   }
//   for (k=0; k<3; k++)	assert(fabs(normalSum[k]) < 1e-5);
//   assert(totalAbsSum > 50);


//   featureKind = ScalarArea;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites,
//                spatialSize,
//                featureKind);

//   totalArea = 0;
//   for (n=0; n<nSpatialSites; n++)	totalArea += features[n];


//   featureKind = Eigenvalues;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites,
//                spatialSize,
//                featureKind);

//   float totalAreaEig = 0;
//   for (n=0; n<nSpatialSites*3; n++)	totalAreaEig += features[n];
//   for (n=0; n<nSpatialSites*3; n++)	assert(features[n] >= -1e-5);
//   assert(fabs(totalAreaEig-totalArea) < 1e-5);

//   featureKind = Eigenvalues;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites1,
//                spatialSize,
//                featureKind);
//   assert(nSpatialSites1 == nSpatialSites);
//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
//   for (n=0; n<nSpatialSites; n++)	assert(features[n] >= -1e-5);

//   featureKind = QFoverSA;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites1,
//                spatialSize,
//                featureKind);
//   assert(nSpatialSites1 == nSpatialSites);
//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
//   for (n=0; n<nSpatialSites; n++)	assert((features[n] <= 1) && (features[n] >= -1));
//   float maxval = -1;
//   float minval = 1;
//   for (n=0; n<nSpatialSites; n++){
//     maxval = fmax(maxval, features[n]);
//     minval = fmin(minval, features[n]);
//   }
//   assert((minval < 1) && (maxval > 0));

//   featureKind = QFoverSA;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites1,
//                spatialSize,
//                featureKind);
//   assert(nSpatialSites1 == nSpatialSites);
//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
//   for (n=0; n<nSpatialSites; n++)	assert((features[n] <= 1) && (features[n] >= -1));
//   maxval = -1;
//   minval = 1;
//   for (n=0; n<nSpatialSites; n++){
//     maxval = fmax(maxval, features[n]);
//     minval = fmin(minval, features[n]);
//   }
//   assert((minval < 1) && (maxval > 0));

//   // Check Euler characteristic
//   double eulerChar;
//   eulerChar = getEulerChar(points, surfaces);
//   assert(fabs(eulerChar-2) < 1e-5);

//   featureKind = AngularDefect;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites1,
//                spatialSize,
//                featureKind);
//   assert(nSpatialSites1 == nSpatialSites);
//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
//   double totalDefect = 0;
//   for (n=0; n<nSpatialSites; n++)	totalDefect += features[n];
//   assert(fabs(eulerChar-totalDefect/(2*M_PI)) < 1e-4);


//   // many features per voxel
//   std::set<enum FeatureKind> featureSet = {Bool, AreaNormal, Quadform, Eigenvalues, QFoverSA, EVoverSA, AngularDefect};
//   assert(nFeaturesPerVoxel_set(featureSet) == 23);
//   get_features_set(points,
//                    surfaces,
//                    grid,
//                    features,
//                    nSpatialSites,
//                    spatialSize,
//                    featureSet);
//   assert(nSpatialSites*nFeaturesPerVoxel_set(featureSet) == (int) features.size());


//   surfaces.clear();
//   curSurf = {0, 1, 2};
//   surfaces.push_back(curSurf);
//   curSurf = {1, 2, 4};
//   surfaces.push_back(curSurf);

//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites,
//                spatialSize,
//                featureKind);

//   assert(nSpatialSites == 36);


//   printf("All tests passed\n");

// }

// void testPerformance(){
//   int n;
//   arma::mat points0;
//   points0 << 0 << 0 << 0 << arma::endr
//           << 0 << 0 << 1 << arma::endr
//           << 0 << 1 << 0 << arma::endr
//           << 1 << 0 << 0 << arma::endr
//           << 0 << 1 << 1 << arma::endr;

//   arma::mat points = points0*50-0.1;

//   std::vector<std::vector<int>> surfaces;
//   std::vector<int> curSurf;
//   curSurf = {0, 1, 2};
//   surfaces.push_back(curSurf);
//   curSurf = {0, 2, 3};
//   surfaces.push_back(curSurf);

//   SparseGrid grid;
//   std::vector<float> features;
//   int nSpatialSites;
//   int spatialSize = 120;
//   std::set<enum FeatureKind> featureSet = {Bool, ScalarArea, AreaNormal, Quadform, Eigenvalues};

//   std::clock_t start;
//   double duration;
//   start = std::clock();

//   for (n = 0; n < 100; n++) {
//     get_features_set(points, surfaces, grid, features, nSpatialSites,
//                      spatialSize, featureSet);
//   }
//   duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
//   printf("Test performance OK; time: %f\n", duration);
// }

// int getMesh(char const *path, arma::mat &points, std::vector<std::vector<int>> &surfaces){
//   // Russian locale may cause decimal point error
//   setlocale(LC_NUMERIC,"C");
//   FILE *meshFile;
//   char line[80];
//   long n, tmp, Nverts, Nfaces;
//   int n0, n1, n2;
//   double x, y, z;
//   meshFile = fopen(path, "rt");
//   if (!meshFile) {
//     printf("File not found!\n");
//     return 1;
//   }
//   fgets(line, 80, meshFile);
//   if (strlen(line) == 4){
//     fgets(line, 80, meshFile);
//     sscanf(line, "%ld %ld", &Nverts, &Nfaces);
//   }else{
//     sscanf(line+3, "%ld %ld", &Nverts, &Nfaces);
//   }

//   points.set_size(Nverts, 3);
//   for (n=0; n<Nverts; n++){
//     fgets(line, 80, meshFile);
//     sscanf(line, "%lf %lf %lf", &x, &y, &z);
//     points(n,0) = x;
//     points(n,1) = y;
//     points(n,2) = z;
//   }
//   std::vector<int> curSurf;
//   for (n=0; n<Nfaces; n++){
//     fgets(line, 80, meshFile);
//     sscanf(line, "%ld %d %d %d", &tmp, &n0, &n1, &n2);
//     curSurf.clear();
//     curSurf.push_back(n0);
//     curSurf.push_back(n1);
//     curSurf.push_back(n2);
//     surfaces.push_back(curSurf);
//   }
//   fclose(meshFile);
//   return 0;
// }

// void scaleMesh(arma::mat &points, int spatialSize){
//   // center and rescale into [-spatialSize,spatialSize]x[-spatialSize,spatialSize]x[-spatialSize,spatialSize]
//   double minA[3], maxA[3], shift[3], scale;
//   unsigned int n, d;
//   for (d=0; d<3; d++){
//     minA[d] = points(0,d);
//     maxA[d] = points(0,d);
//     for (n=0; n<points.n_rows; n++){
//       minA[d] = fmin(minA[d], points(n,d));
//       maxA[d] = fmax(maxA[d], points(n,d));
//     }
//     shift[d] = 0.5*(minA[d]+maxA[d]);
//   }

//   scale = spatialSize/(maxA[0]-minA[0])/1.2;
//   for (d=1; d<3; d++){
//     scale = fmin(scale, spatialSize/(maxA[d]-minA[d])/1.2);
//   }
//   for (d=0; d<3; d++){
//     for (n=0; n<points.n_rows; n++){
//       points(n,d) -= shift[d];
//       points(n,d) *= scale;
//       assert(points(n,d) > -spatialSize/2);
//       assert(points(n,d) < spatialSize/2);
//     }
//   }

// }

// void testPerformanceModelNet(){
//   int n;
//   arma::mat points;
//   std::vector<std::vector<int>> surfaces;
//   int res;
//   //const char path[] = "/home/dmitry.yarotsky/current_work/3Dshapes/ModelNet10/bathtub/train/bathtub_0003.off";
//   //const char path[] = "/home/dmitry.yarotsky/current_work/3Dshapes/ESB/Flat-Thin Wallcomponents/Back Doors/backdoor5.off";
//   const char path[] = "/home/dmitry.yarotsky/current_work/3Dshapes/ModelNet10/chair/train/chair_0643.off";
//   puts(path);

//   res = getMesh(path, points, surfaces);
//   if (!(res==0)){
//     return;
//   }
//   printf("Mesh loaded OK\n");
//   printf("Nverts, Nfaces: %d, %d\n", points.n_rows, (int) surfaces.size());

//   SparseGrid grid;
//   std::vector<float> features;
//   int nSpatialSites, nSpatialSites1;
//   int spatialSize = 120;
//   scaleMesh(points, spatialSize);
//   printf("Mesh rescaled OK\n");

//   enum FeatureKind featureKind = Bool;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites,
//                spatialSize,
//                featureKind);

//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
//   for (n=0; n<nSpatialSites; n++)	assert(features[n] == 1.f);


//   featureKind = ScalarArea;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites1,
//                spatialSize,
//                featureKind);
//   assert(nSpatialSites1 == nSpatialSites);
//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
//   for (n=0; n<nSpatialSites; n++)	assert(features[n] >= 0);
//   for (n=0; n<nSpatialSites; n++)	assert(features[n] <= 100);


//   featureKind = Quadform;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites1,
//                spatialSize,
//                featureKind);
//   assert(nSpatialSites1 == nSpatialSites);
//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());

//   for (n=0; n<nSpatialSites; n++)	{
//     assert(features[6*n] >= 0);
//     assert(features[6*n+1] >= 0);
//     assert(features[6*n+2] >= 0);
//     assert(features[6*n]*features[6*n+1] >= features[6*n+3]*features[6*n+3]-1e-5);
//   }

//   featureKind = Eigenvalues;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites1,
//                spatialSize,
//                featureKind);
//   assert(nSpatialSites1 == nSpatialSites);
//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
//   for (n=0; n<nSpatialSites; n++)	assert(features[n] >= -1e-5);

//   featureKind = QFoverSA;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites1,
//                spatialSize,
//                featureKind);
//   assert(nSpatialSites1 == nSpatialSites);
//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
//   for (n=0; n<nSpatialSites; n++)	assert((features[n] <= 1) && (features[n] >= -1));
//   float maxval = -1;
//   float minval = 1;
//   for (n=0; n<nSpatialSites; n++){
//     maxval = fmax(maxval, features[n]);
//     minval = fmin(minval, features[n]);
//   }
//   assert((minval < 1) && (maxval > 0));

//   featureKind = EVoverSA;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites1,
//                spatialSize,
//                featureKind);
//   assert(nSpatialSites1 == nSpatialSites);
//   assert(nSpatialSites == (int) grid.mp.size());
//   assert(nSpatialSites*nFeaturesPerVoxel(featureKind) == (int) features.size());
//   for (n=0; n<nSpatialSites; n++)	assert((features[n] <= 1) && (features[n] >= -1e-7));
//   maxval = -1;
//   minval = 1;
//   for (n=0; n<nSpatialSites; n++){
//     maxval = fmax(maxval, features[n]);
//     minval = fmin(minval, features[n]);
//   }
//   assert((minval < 1) && (maxval > 0));

//   std::set<enum FeatureKind> featureSet = {Bool, ScalarArea, AreaNormal, Quadform, Eigenvalues, QFoverSA, EVoverSA, AngularDefect};
//   int nFeaturesPV = nFeaturesPerVoxel_set(featureSet);
//   get_features_set(points,
//                    surfaces,
//                    grid,
//                    features,
//                    nSpatialSites,
//                    spatialSize,
//                    featureSet);
//   printf("nSpatialSites: %d\n", nSpatialSites);
//   float area, totalArea;
//   totalArea = 0;
//   for (n=0; n<nSpatialSites; n++)	{
//     assert(features[nFeaturesPV*n]*(features[nFeaturesPV*n]-1) == 0);
//     area = features[nFeaturesPV*n+1];
//     assert(area >= 0);
//     totalArea += area;
//     assert(fabs(features[nFeaturesPV*n+5]+features[nFeaturesPV*n+6]+features[nFeaturesPV*n+7]-area) <= 1e-5);
//     assert(fabs(features[nFeaturesPV*n+11]+features[nFeaturesPV*n+12]+features[nFeaturesPV*n+13]-area) <= 1e-5);
//   }
//   printf("Total area: %f\n", totalArea);

//   printf("Unit tests passed\n");

//   std::clock_t start;
//   double duration;
//   start = std::clock();
//   int Ncalls = 10;
//   for (n = 0; n < Ncalls; n++) {
//     get_features_set(points, surfaces, grid, features, nSpatialSites,
//                      spatialSize, featureSet);
//   }
//   duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
//   printf("Performance test with %d calls to get_features_set() OK; time: %f\n", Ncalls, duration);
// }

// void testEuler(){
//   int n;
//   arma::mat points;
//   std::vector<std::vector<int>> surfaces;
//   int res;
//   //const char path[] = "/home/dmitry.yarotsky/current_work/3Dshapes/ESB/Rectangular-Cubic Prism/Prismatic Stock/RCR32_M_.off";
//   //const char path[] = "/home/dmitry.yarotsky/current_work/3Dshapes/ModelNet10/toilet/train/toilet_0339.off";
//   const char path[] = "/home/dmitry.yarotsky/current_work/3Dshapes/ModelNet10/chair/train/chair_0643.off";
//   //const char path[] = "/home/dmitry.yarotsky/current_work/3Dshapes/ESB/Flat-Thin Wallcomponents/Contact Switches/schmersal_st_14_b5.off";
//   puts(path);

//   res = getMesh(path, points, surfaces);
//   if (!(res==0)){
//     return;
//   }
//   printf("Mesh loaded OK\n");
//   printf("Nverts, Nfaces: %d, %d\n", points.n_rows, (int) surfaces.size());

//   double eulerChar;
//   eulerChar = getEulerChar(points, surfaces);
//   printf("Euler characteristic: %f\n", eulerChar);

//   SparseGrid grid;
//   std::vector<float> features;
//   int nSpatialSites;
//   int spatialSize = 20;
//   scaleMesh(points, spatialSize);
//   printf("Mesh rescaled OK\n");

//   enum FeatureKind featureKind = AngularDefect;
//   get_features(points,
//                surfaces,
//                grid,
//                features,
//                nSpatialSites,
//                spatialSize,
//                featureKind);

//   double totalDefect = 0;
//   for (n=0; n<nSpatialSites; n++)	totalDefect += features[n];
//   assert(fabs(totalDefect/(2*M_PI)-eulerChar) < 1e-4);
// }
