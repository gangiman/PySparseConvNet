#pragma once
#include "Picture.h"
#include "Rng.h"
#include <armadillo>
#include <string>
#include <vector>

class VoxelPicture : public Picture {
private:
  std::vector<int> non_empty_indices;
  std::vector<float> voxel_features;
public:
  int renderSize;
  int n_features;
  VoxelPicture(const std::vector<std::vector<int>>& indices,
                           const std::vector<std::vector<float>>& input_features,
                           int renderSize, int label, int n_features);

  virtual ~VoxelPicture() {
    non_empty_indices.clear();
    voxel_features.clear();
  }
  void loadPicture();
  void unloadPicture() override;
  void codifyInputData(SparseGrid &grid, std::vector<float> &features, int &nSpatialSites, int spatialSize);
  Picture *distort(RNG &rng, batchType type = TRAINBATCH);
};
