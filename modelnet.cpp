//#include "SparseConvNetCUDA.h"
#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetModelNet.h"

int epoch = 0;
int cudaDevice = -1;
int batchSize = 10;

int main(int lenArgs, char *args[]) {
  std::string baseName = "weights/ModelNet";
  int fold = 0;
  if (lenArgs > 1)
    fold = atoi(args[1]);
  std::cout << "Fold: " << fold << std::endl;
  SpatiallySparseDataset trainSet = ModelNetDataSet(40, 6, fold, true);
  trainSet.summary();
//  trainSet.repeatSamples(10);
  SpatiallySparseDataset testSet = ModelNetDataSet(40, 6, fold, false);
  testSet.summary();

  DeepC2 cnn(3, 5, 32, VLEAKYRELU, trainSet.nFeatures, trainSet.nClasses, 0.0f,
             cudaDevice);
  if (epoch > 0)
    cnn.loadWeights(baseName, epoch);
  for (epoch++; epoch <= 100 * 2; epoch++) {
    std::cout << "epoch:" << epoch << ": " << std::flush;
    cnn.processDataset(trainSet, batchSize, 0.003 * exp(-0.05 / 2 * epoch));
    if (epoch % 20 == 0) {
      cnn.saveWeights(baseName, epoch);
      cnn.processDatasetRepeatTest(testSet, batchSize, 3);
    }
  }
}
