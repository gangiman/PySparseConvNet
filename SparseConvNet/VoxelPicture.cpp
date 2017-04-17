#include "VoxelPicture.h"
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>


//VoxelPicture(vector[vector[int]] indices, vector[int] features, int spatial_size)
VoxelPicture::VoxelPicture(const std::vector<std::vector<int>>& indices,
                           const std::vector<std::vector<float>>& input_features,
                           int renderSize, int label, int n_features)
    : Picture(label), renderSize(renderSize), n_features(n_features)
{
    // assert(indices.size() == features.size());
    // number of features is multiple of number of active voxels
    assert(input_features.size() % indices.size() == 0);
    assert(n_features == input_features.size() / indices.size());

    for (size_t feature = 0; feature < n_features; ++feature) {
      voxel_features.push_back(0);
    }

    for (int i{}; i < indices.size(); ++i) {
        assert(indices[i].size() == 3);
        int n = indices[i][0] * renderSize * renderSize + indices[i][1] * renderSize + indices[i][2];
        non_empty_indices.push_back(n);
        for (int feature_id = 0; feature_id < n_features; ++feature_id) {
            voxel_features.push_back(input_features[i][feature_id]);
          }
    }
    is_loaded = true;
}

void VoxelPicture::loadPicture() {
  is_loaded = true;
}

void VoxelPicture::unloadPicture() {
  // non_empty_indices.resize(0);
  // voxel_features.resize(0);
  is_loaded = false;
}


void VoxelPicture::codifyInputData(SparseGrid &grid,
                                   std::vector<float> &features,
                                   int &nSpatialSites,
                                   int spatialSize) {
  nSpatialSites = 0;
  grid.backgroundCol = nSpatialSites++;
  for (int i = 0; i < non_empty_indices.size(); ++i) {

    int n = non_empty_indices[i];
    if (grid.mp.find(n) == grid.mp.end()) {
      grid.mp[n] = nSpatialSites++;
      // features.push_back(1); // NOTE: here we can push back vector of features
    }
  }
  features = voxel_features;
}

Picture *VoxelPicture::distort(RNG &rng, batchType type) {
  VoxelPicture *pic = new VoxelPicture(*this);
  return pic;
}
