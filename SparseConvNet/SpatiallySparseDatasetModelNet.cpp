// Try adding normal vectors to input data.

// Data from http://www.icst.pku.edu.cn/zlian/shrec15-non-rigid/index.htm
// 50 classes, 24 exemplars per class: alien ants armadillo bird1 bird2 camel
// cat centaur twoballs dinosaur dog1 dog2 glasses gorilla hand horse lamp paper
// man octopus pliers rabbit santa scissor shark snake spider dino_ske flamingo
// woman Aligator Bull Chick Deer Dragon Elephant Frog Giraffe Kangaroo Mermaid
// Mouse Nunchaku MantaRay Ring Robot Sumotori Tortoise Watch Weedle Woodman

#include "SpatiallySparseDatasetModelNet.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "OpenCVPicture.h"
#include "Off3DFormatPicture.h"

//#include <sys/types.h>
#include <dirent.h>
#include <errno.h>


std::vector<std::string> getdir (std::string dir)
{
  std::vector<std::string> files = std::vector<std::string>();
  DIR *dp;
  struct dirent *dirp;
  if((dp = opendir(dir.c_str())) == NULL) {
    std::cout << "Error(" << errno << ") opening " << dir << std::endl;
    return files;
  }
  char dot = '.';
  while ((dirp = readdir(dp)) != NULL) {
    std::string entery = std::string(dirp->d_name);
    if (entery.at(0) != dot){
      files.push_back(entery);
    }
  }
  closedir(dp);
  return files;
}

SpatiallySparseDataset ModelNetDataSet(int renderSize, int kFold, int fold, batchType batch_type) {
  SpatiallySparseDataset dataset;
  dataset.type = batch_type;
  std::string mode;
  int max_pictures_per_class;
  if (batch_type == TRAINBATCH) {
    dataset.name = "ModelNet (Train subset)";
    mode = std::string("train");
    max_pictures_per_class = 80;
  } else if (batch_type == TESTBATCH) {
    dataset.name = "ModelNet (Validation subset)";
    mode = std::string("test");
    max_pictures_per_class = 20;
  } else {}
  dataset.nFeatures = 1;
  std::string basedir = std::string("/media/toshiba/shape_retrieval_datasets/ModelNet/ModelNet40/");
  std::vector<std::string> classes = getdir(basedir);
  dataset.nClasses = classes.size();
  std::cout << "Number of classes: " << dataset.nClasses << std::endl;
  sort(classes.begin(), classes.end());
  for (unsigned int class_id = 0;class_id < classes.size();class_id++) {
    std::string class_dir = basedir + classes[class_id] + std::string("/") + mode;
    std::vector<std::string> files = getdir(class_dir);
    for (int i = 0; i < std::min(max_pictures_per_class, (int)files.size()); ++i) {
      std::string filename = class_dir + std::string("/") + files[i];
      dataset.pictures.push_back(
              new OffSurfaceModelPicture(filename, renderSize, class_id));
    }
  }

  return dataset;
};
