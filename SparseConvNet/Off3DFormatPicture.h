#pragma once
#include "Picture.h"
#include "Rng.h"
#include <armadillo>
#include <vector>
#include <string>
#include <set>

class OffSurfaceModelPicture : public Picture {
private:
  arma::mat points;
  std::vector<std::vector<int>>
      surfaces; // Will assume all surfaces are triangles for now
public:
  int renderSize;
  std::string picture_path;
  std::set<FeatureKind> feature_kind;
  bool is_loaded;
  OffSurfaceModelPicture(std::string filename, int renderSize, int label_ = -1, std::set<FeatureKind> feature_kind={Bool});
  ~OffSurfaceModelPicture();
  void loadPicture();
  void normalize(); // Fit centrally in the cube [-scale_n/2,scale_n/2]^3
  void random_rotation(RNG &rng);
  void jiggle(RNG &rng, float alpha);
  void affineTransform(RNG &rng, float alpha);
  void codifyInputData(SparseGrid &grid, std::vector<float> &features,
                       int &nSpatialSites, int spatialSize);
  Picture *distort(RNG &rng, batchType type = TRAINBATCH);
};
