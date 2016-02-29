#pragma once
#include "SpatiallySparseDataset.h"
#include <iostream>

SpatiallySparseDataset ModelNetDataSet(int renderSize, int kFold, int fold, bool is_train);
