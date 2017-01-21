#pragma once
#include "Picture.h"
#include "Rng.h"
#include <armadillo>
#include <string>
#include <vector>

class VoxelPicture : public Picture {
private:
  arma::mat points;
  std::vector<int> non_empty_indices;
  std::vector<float> voxel_features;
public:
  int renderSize;
  int n_features;
  std::string picture_path;
  VoxelPicture(std::vector<float>& voxels, int renderSize=0, int label_=-1, int n_features=1);
  virtual ~VoxelPicture() {
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
