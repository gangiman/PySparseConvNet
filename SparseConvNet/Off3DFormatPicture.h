#pragma once
#include "Picture.h"
#include "Rng.h"
#include <armadillo>
#include <string>
#include <vector>

class OffSurfaceModelPicture : public Picture {
private:
  arma::mat points;
  std::vector<std::vector<int>>
      surfaces; // Will assume all surfaces are triangles for now

  std::mutex distort_mtx;
public:
  int renderSize;
  std::string picture_path;

  OffSurfaceModelPicture(const OffSurfaceModelPicture& other)
    : points(other.points)
    , surfaces(other.surfaces)
    , renderSize(other.renderSize)
    , picture_path(other.picture_path)
  {}
  OffSurfaceModelPicture(std::string filename, int renderSize, int label_ = -1);
  virtual ~OffSurfaceModelPicture() {
    points.reset();
  }
  void loadPicture();
  void unloadPicture() override;
  void normalize(); // Fit centrally in the cube [-scale_n/2,scale_n/2]^3
  void random_rotation(RNG &rng);
  void jiggle(RNG &rng, float alpha);
  void affineTransform(RNG &rng, float alpha);
  void codifyInputData(SparseGrid &grid, std::vector<float> &features, int &nSpatialSites, int spatialSize);
  Picture *distort(RNG &rng, batchType type = TRAINBATCH);
};
