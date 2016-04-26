#include <armadillo>
#include <vector>
#include "SparseGrid.h"
#include <set>
#include "types.h"

void get_features_set(arma::mat points,
                      std::vector<std::vector<int>> surfaces,
                      SparseGrid &grid,
                      std::vector<float> &features,
                      int &nSpatialSites,
                      int spatialSize,
                      std::set<enum FeatureKind> featureSet);

int nFeaturesPerVoxel_set(std::set<enum FeatureKind> featureSet);