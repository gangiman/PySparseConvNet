#include "VoxelPicture.h"
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>

// int mapToGridOFF(float coord, int inputFieldSize) {
//   return std::max(
//       0, std::min(inputFieldSize - 1, (int)(coord + 0.5 * inputFieldSize)));
// }


int mapToGridVoxels(float coord, int inputFieldSize) {
  return std::max(
      0, std::min(inputFieldSize - 1, (int)(coord)));
}

// Draw triangle with vertices {a, a+u, a+v} on grid
// For every entry in grid (except the null vector -1) add a +1 to features
// vector.
// void drawTriangleOFF(SparseGrid &grid, int inputFieldSize,
//                      std::vector<float> &features, int &nSpatialSites, float a0,
//                      float a1, float a2, float u0, float u1, float u2, float v0,
//                      float v1, float v2) {
//   float base = powf(u0 * u0 + u1 * u1 + u2 * u2, 0.5);
//   u0 /= base;
//   u1 /= base;
//   u2 /= base;                                 // scale u to a unit vector
//   float offset = u0 * v0 + u1 * v1 + u2 * v2; // u dot v
//   v0 -= offset * u0;
//   v1 -= offset * u1;
//   v2 -= offset * u2; // make v orthogonal to u
//   float height = powf(v0 * v0 + v1 * v1 + v2 * v2, 0.5);
//   v0 /= height;
//   v1 /= height;
//   v2 /= height; // scale v to be a unit vector
//   // u and v are now orthogonal
//   // The triangle now has points {a, a+base*u, a+offset*u+height*v}

//   for (float h = 0; h <= height; h = std::min(h + 1, height) + (h == height)) {
//     float l = base * (1 - h / height);
//     for (float b = 0; b <= l; b = std::min(b + 1, l) + (b == l)) {
//       float p0 = a0 + (b + offset * h / height) * u0 + h * v0,
//             p1 = a1 + (b + offset * h / height) * u1 + h * v1,
//             p2 = a2 + (b + offset * h / height) * u2 + h * v2;
//       int n =
//           mapToGridOFF(p0, inputFieldSize) * inputFieldSize * inputFieldSize +
//           mapToGridOFF(p1, inputFieldSize) * inputFieldSize +
//           mapToGridOFF(p2, inputFieldSize);
//       if (grid.mp.find(n) == grid.mp.end()) {
//         grid.mp[n] = nSpatialSites++;
//         features.push_back(1); // NOTE: here we can push back vector of features
//       }
//     }
//   }
// }


//VoxelPicture(vector[vector[int]] indices, vector[int] features, int spatial_size)
VoxelPicture::VoxelPicture(const std::vector<std::vector<int>>& indices,
                           const std::vector<int>& features, int spatial_size)
    : Picture(-1)
    , renderSize(-1)
    , n_features(-1)
{
    assert(indices.size() == features.size());

    for (int i{}; i < indices.size(); ++i) {
        assert(indices[i].size() == 3);
        int n = indices[i][0] * renderSize * renderSize + indices[i][1] * renderSize + indices[i][2];
    }
}

VoxelPicture::VoxelPicture(std::vector<float>& voxels,
                           int renderSize, int label, int n_features)
    : Picture(label), renderSize(renderSize), n_features(n_features) {

  // insert background features
  for (size_t feature = 0; feature < n_features; ++feature) {
    voxel_features.push_back(0);
  }

  for (size_t i = 0; i < renderSize; ++i) {
    for (size_t j = 0; j < renderSize; ++j) {
      for (size_t k = 0; k < renderSize; ++k) {
        int n = i * renderSize * renderSize + j * renderSize + k;
        bool filled_voxel = false;
        for (size_t feature = 0; feature < n_features; ++feature) {
          if (voxels[n * n_features + feature] >= 1e-2) {
            filled_voxel = true;
            break;
          }
        }
        if (filled_voxel) {
          non_empty_indices.push_back(n);
          for (size_t feature = 0; feature < n_features; ++feature) {
            voxel_features.push_back(voxels[n * n_features + feature]);
          }
        }
      }
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

void VoxelPicture::normalize() { // Fit centrally in the cube
                                 // [-renderSize/2,renderSize/2]^3
}

void VoxelPicture::random_rotation(RNG &rng) {
  // arma::mat L, Q, R;
  // L.set_size(3, 3);
  // for (int i = 0; i < 3; i++)
  //   for (int j = 0; j < 3; j++)
  //     L(i, j) = rng.uniform();
  // arma::qr(Q, R, L);
  // points = points * Q;
}

void VoxelPicture::jiggle(RNG &rng, float alpha) {
  // for (int i = 0; i < 3; i++)
  //   points.col(i) += renderSize * rng.uniform(-alpha, alpha);
}

void VoxelPicture::affineTransform(RNG &rng, float alpha) {
  // arma::mat L = arma::eye<arma::mat>(3, 3);
  // for (int i = 0; i < 3; i++)
  //   for (int j = 0; j < 3; j++)
  //     L(i, j) += rng.uniform(-alpha, alpha);
  // points = points * L;
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
  if (type != UNLABELEDBATCH)
  {
    pic->random_rotation(rng);
    // pic->normalize();
    // if (type == TRAINBATCH) {
    //   pic->affineTransform(rng, 0.2);
    //   pic->jiggle(rng, 0.2);
    // }
  // }else{
    // pic->normalize();
  }
  return pic;
}
