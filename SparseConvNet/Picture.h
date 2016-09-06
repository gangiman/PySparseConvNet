#pragma once
#include <string>
#include <vector>
#include "SparseGrid.h"
#include "Rng.h"
#include "types.h"

class Picture {
public:
  virtual void codifyInputData(SparseGrid &grid, std::vector<float> &features,
                               int &nSpatialSites, int spatialSize) = 0;
  virtual std::shared_ptr<Picture> distort(RNG &rng, batchType type) { return std::shared_ptr<Picture>(this); }
  virtual std::string identify();
  virtual void loadPicture() = 0;
  bool is_loaded;
  int label; //-1 for unknown
  Picture(int label = -1);
  virtual ~Picture();
};
