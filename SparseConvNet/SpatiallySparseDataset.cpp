#include "SpatiallySparseDataset.h"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <vector>

void SpatiallySparseDataset::summary() {
  std::cout << "Name:            " << name << std::endl;
  std::cout << "Type:            " << batchTypeNames[type] << std::endl;
  std::cout << "nFeatures:       " << nFeatures << std::endl;
  std::cout << "nPictures:       " << pictures.size() << std::endl;
  std::cout << "nClasses:        " << nClasses << std::endl;
  if (type != UNLABELEDBATCH){
    std::vector<int> count(nClasses);
    for (int _i=0;_i<pictures.size();_i++)
      count[ real_pictures[pictures[_i]]->label ]++;

    std::cout << "nPictures/class: ";
    for (int _i=0;_i<count.size();_i++)
      std::cout << count[_i] << " ";
    std::cout << std::endl;
  }
}
SpatiallySparseDataset SpatiallySparseDataset::extractValidationSet(float p) {
  SpatiallySparseDataset val;
  val.name = name + std::string(" Validation set");
  name = name + std::string(" minus Validation set");
  val.nClasses = nClasses;
  val.nFeatures = nFeatures;
  val.type = TESTBATCH;
  std::mt19937 gen(123);
  std::shuffle(pictures.begin(), pictures.end(), gen);
  int size = pictures.size() * p;
  for (; size > 0; size--) {
    val.pictures.push_back(pictures.back());
    pictures.pop_back();
  }
  return val;
}

// Assume there are at least n of each class in the dataset
SpatiallySparseDataset SpatiallySparseDataset::balancedSubset(int n) {
  SpatiallySparseDataset bs;
  bs.name = name + std::string(" subset");
  bs.nFeatures = nFeatures;
  bs.nClasses = nClasses;
  bs.type = type;
  auto permutation = rng.permutation(pictures.size());
  std::vector<int> count(nClasses);
  int classesDone = 0;
  for (unsigned int i = 0; i < pictures.size() and classesDone < nClasses;
       i++) {
    auto pic = pictures[permutation[i]];
    auto real_pic = real_pictures[pictures[permutation[i]]];
    if (count[real_pic->label]++ < n)
      bs.pictures.push_back(pic);
    if (count[real_pic->label] == n)
      classesDone++;
  }
  return bs;
}

SpatiallySparseDataset SpatiallySparseDataset::subset(int n) {
  SpatiallySparseDataset subset;
  subset.name = name + std::string(" subset");
  subset.nFeatures = nFeatures;
  subset.nClasses = nClasses;
  subset.type = type;
  auto pick = rng.NchooseM(pictures.size(), n);
  for (auto i : pick) {
    subset.pictures.push_back(pictures[i]);
  }
  return subset;
}

void SpatiallySparseDataset::shuffle() {
  std::shuffle(pictures.begin(), pictures.end(), rng.gen);
}
void SpatiallySparseDataset::repeatSamples(int reps) {
  int s = pictures.size();
  for (int i = 1; i < reps; ++i)
    for (int j = 0; j < s; ++j)
      pictures.push_back(pictures[j]);
}

// http://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
std::vector<std::string> globVector(const std::string &pattern) {
  glob_t glob_result;
  glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  std::vector<std::string> files;
  for (unsigned int i = 0; i < glob_result.gl_pathc; ++i) {
    files.push_back(std::string(glob_result.gl_pathv[i]));
  }
  globfree(&glob_result);
  std::sort(files.begin(), files.end());
  return files;
}

std::shared_ptr<Picture> get_fucking_picture(const int &i) {
    return std::shared_ptr<Picture>();
}

// Usage: std::vector<std::string> files = globVector("./*");
