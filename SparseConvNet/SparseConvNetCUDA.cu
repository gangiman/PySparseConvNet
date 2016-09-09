#include "SparseConvNetCUDA.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <sstream>
#include "utilities.h"
#include "SigmoidLayer.h"
#include "NetworkInNetworkLayer.h"
#include "NetworkInNetworkPReLULayer.h"
#include "ConvolutionalLayer.h"
#include "ReallyConvolutionalLayer.h"
#include "ConvolutionalTriangularLayer.h"
#include "MaxPoolingLayer.h"
#include "MaxPoolingTriangularLayer.h"
#include "TerminalPoolingLayer.h"
#include "IndexLearnerLayer.h"
#include "SoftmaxClassifier.h"
#include "BatchProducer.h"
#include "SpatiallySparseDataset.h"
#include <chrono>

SparseConvNetCUDA::SparseConvNetCUDA(int dimension, int nInputFeatures,
                                     int nClasses, int pciBusID, int nTop,
                                     int nBatchProducerThreads)
    : deviceID(initializeGPU(pciBusID)), dimension(dimension),
      nInputFeatures(nInputFeatures), nClasses(nClasses), nTop(nTop),
      nBatchProducerThreads(nBatchProducerThreads) {
  assert(nBatchProducerThreads <= N_MAX_BATCH_PRODUCER_THREADS);
  /* std::cout << "Sparse CNN - dimension=" << dimension
            << " nInputFeatures=" << nInputFeatures << " nClasses=" << nClasses
            << " nThreads=" << nBatchProducerThreads
            << std::endl; */
  nOutputFeatures = nInputFeatures;
  // Set up a pool of SpatiallySparseBatches
  for (int c = 0; c < nBatchProducerThreads; c++) {
    initialSubInterfaces.push_back(new SpatiallySparseBatchSubInterface());
    batchPool.emplace_back(initialSubInterfaces.back());
  }
  cublasError(cublasCreate(&cublasHandle), __FILE__, __LINE__);
  cublasError(cublasSetStream(cublasHandle, memStream.stream));
}
SparseConvNetCUDA::~SparseConvNetCUDA() {
  for (auto p : initialSubInterfaces)
    delete p;
  for (auto p : sharedSubInterfaces)
    delete p;
  cublasError(cublasDestroy(cublasHandle), __FILE__, __LINE__);
}
void SparseConvNetCUDA::addLearntLayer(int nFeatures,
                                       ActivationFunction activationFn,
                                       float dropout, float alpha) {
  if (activationFn != SOFTMAX)
    nFeatures = std::max(KERNELBLOCKSIZE, intRound(nFeatures, KERNELBLOCKSIZE));
  if (dropout > 0)
    dropout = 1 -
              (intRound(nFeatures * (1 - dropout), KERNELBLOCKSIZE) + 0.01f) *
                  1.0f / nFeatures;
  if (dropout < 0)
    dropout = 0;
  /* std::cout << layers.size() << ":"; */
  if (activationFn == PRELU)
    layers.push_back(new NetworkInNetworkPReLULayer(
        memStream, cublasHandle, nOutputFeatures, nFeatures, dropout, alpha));
  else
    layers.push_back(new NetworkInNetworkLayer(memStream, cublasHandle,
                                               nOutputFeatures, nFeatures,
                                               dropout, activationFn, alpha));
  nOutputFeatures = nFeatures;
}
void SparseConvNetCUDA::addNetworkInNetworkLayer(
    int nFeatures, ActivationFunction activationFn, float dropout) {
  addLearntLayer(nFeatures, activationFn, dropout, 1.0f);
}
void SparseConvNetCUDA::addConvolutionalLayer(int nFeatures, int filterSize,
                                              int filterStride,
                                              ActivationFunction activationFn,
                                              float dropout,
                                              int minActiveInputs,
                                              float poolingToFollow) {
  if (false and layers.size() == 0) { // Use for 0-th layer??
    /* std::cout << layers.size() << ":"; */
    layers.push_back(new ReallyConvolutionalLayer(
        memStream, nOutputFeatures, nFeatures, filterSize, filterStride,
        dimension, activationFn, dropout, minActiveInputs, poolingToFollow));
    nOutputFeatures = nFeatures;
  } else {
    if (filterSize > 1) {
      /* std::cout << layers.size() << ":"; */
      layers.push_back(
          new ConvolutionalLayer(memStream, filterSize, filterStride, dimension,
                                 nOutputFeatures, minActiveInputs));
      nOutputFeatures *= ipow(filterSize, dimension);
    }
    addLearntLayer(nFeatures, activationFn, dropout,
                   powf(filterSize * 1.0 / filterStride / poolingToFollow, 2));
  }
}
void SparseConvNetCUDA::addLeNetLayerMP(int nFeatures, int filterSize,
                                        int filterStride, int poolSize,
                                        int poolStride,
                                        ActivationFunction activationFn,
                                        float dropout, int minActiveInputs) {
  addConvolutionalLayer(nFeatures, filterSize, filterStride, activationFn,
                        dropout, minActiveInputs, poolSize);
  if (poolSize > 1) {
    /* std::cout << layers.size() << ":"; */
    layers.push_back(
        new MaxPoolingLayer(memStream, poolSize, poolStride, dimension));
  }
}
void SparseConvNetCUDA::addLeNetLayerROFMP(int nFeatures, int filterSize,
                                           int filterStride, int poolSize,
                                           float fmpShrink,
                                           ActivationFunction activationFn,
                                           float dropout, int minActiveInputs) {
  addConvolutionalLayer(nFeatures, filterSize, filterStride, activationFn,
                        dropout, minActiveInputs, fmpShrink);
  if (fmpShrink > 1) {
    /* std::cout << layers.size() << ":"; */
    layers.push_back(new RandomOverlappingFractionalMaxPoolingLayer(
        memStream, poolSize, fmpShrink, dimension));
  }
}
void SparseConvNetCUDA::addLeNetLayerPOFMP(int nFeatures, int filterSize,
                                           int filterStride, int poolSize,
                                           float fmpShrink,
                                           ActivationFunction activationFn,
                                           float dropout, int minActiveInputs) {
  addConvolutionalLayer(nFeatures, filterSize, filterStride, activationFn,
                        dropout, minActiveInputs, fmpShrink);
  if (fmpShrink > 1) {
    /* std::cout << layers.size() << ":"; */
    layers.push_back(new PseudorandomOverlappingFractionalMaxPoolingLayer(
        memStream, poolSize, fmpShrink, dimension));
  }
}
void SparseConvNetCUDA::addLeNetLayerRDFMP(int nFeatures, int filterSize,
                                           int filterStride, int poolSize,
                                           float fmpShrink,
                                           ActivationFunction activationFn,
                                           float dropout, int minActiveInputs) {
  addConvolutionalLayer(nFeatures, filterSize, filterStride, activationFn,
                        dropout, minActiveInputs, fmpShrink);
  if (fmpShrink > 1) {
    /* std::cout << layers.size() << ":"; */
    layers.push_back(new RandomNonOverlappingFractionalMaxPoolingLayer(
        memStream, poolSize, fmpShrink, dimension));
  }
}
void SparseConvNetCUDA::addLeNetLayerPDFMP(int nFeatures, int filterSize,
                                           int filterStride, int poolSize,
                                           float fmpShrink,
                                           ActivationFunction activationFn,
                                           float dropout, int minActiveInputs) {
  addConvolutionalLayer(nFeatures, filterSize, filterStride, activationFn,
                        dropout, minActiveInputs, fmpShrink);
  if (fmpShrink > 1) {
    /* std::cout << layers.size() << ":"; */
    layers.push_back(new PseudorandomNonOverlappingFractionalMaxPoolingLayer(
        memStream, poolSize, fmpShrink, dimension));
  }
}

void SparseConvNetCUDA::addTriangularConvolutionalLayer(
    int nFeatures, int filterSize, int filterStride,
    ActivationFunction activationFn, float dropout, int minActiveInputs,
    float poolingToFollow) {
  if (filterSize > 1) {
    /* std::cout << layers.size() << ":"; */
    layers.push_back(new ConvolutionalTriangularLayer(
        memStream, filterSize, filterStride, dimension, nOutputFeatures,
        minActiveInputs));
    nOutputFeatures *= triangleSize(filterSize, dimension);
  }
  addLearntLayer(nFeatures, activationFn, dropout,
                 powf(filterSize * 1.0 / filterStride / poolingToFollow, 2));
}
void SparseConvNetCUDA::addTriangularLeNetLayerMP(
    int nFeatures, int filterSize, int filterStride, int poolSize,
    int poolStride, ActivationFunction activationFn, float dropout,
    int minActiveInputs) {
  addTriangularConvolutionalLayer(nFeatures, filterSize, filterStride,
                                  activationFn, dropout, poolSize,
                                  minActiveInputs);
  if (poolSize > 1) {
    /* std::cout << layers.size() << ":"; */
    layers.push_back(new MaxPoolingTriangularLayer(memStream, poolSize,
                                                   poolStride, dimension));
  }
}

void SparseConvNetCUDA::addTerminalPoolingLayer(int poolSize, int S) {
  /* std::cout << layers.size() << ":"; */
  layers.push_back(new TerminalPoolingLayer(memStream, poolSize, S));
}

int SparseConvNetCUDA::computeInputSpatialSize(int outputSpatialSize) {
  inputSpatialSize = outputSpatialSize;
  for (int i = layers.size() - 1; i >= 0; i--) {
    inputSpatialSize = layers[i]->calculateInputSpatialSize(inputSpatialSize);
  }
  return inputSpatialSize;
};

void SparseConvNetCUDA::addSoftmaxLayer() {
  addLearntLayer(nClasses, SOFTMAX, 0.0f, 1);
  inputSpatialSize = 1;
  /* std::cout << "Spatially sparse CNN with layer sizes: " << inputSpatialSize; */
  for (int i = layers.size() - 1; i >= 0; i--) {
    inputSpatialSize = layers[i]->calculateInputSpatialSize(inputSpatialSize);
  }
  /* std::cout << std::endl; */
  /* std::cout << "Input-field dimensions = " << inputSpatialSize; */
  // for (int i = 1; i < dimension; ++i)
    /* std::cout << "x" << inputSpatialSize; */
  /* std::cout << std::endl; */
}
void SparseConvNetCUDA::addIndexLearnerLayer() {
  /* std::cout << layers.size() << ":"; */
  layers.push_back(new IndexLearnerLayer(memStream, cublasHandle,
                                         nOutputFeatures, nClasses));
  /* std::cout << "Index Learner " << nOutputFeatures << "-> " << nClasses
            << std::endl; */
  nOutputFeatures = nClasses; // "nClasses"=trainingSet.pictures.size()
  inputSpatialSize = 1;
  for (int i = layers.size() - 1; i >= 0; i--) {
    inputSpatialSize = layers[i]->calculateInputSpatialSize(inputSpatialSize);
  }
  /* std::cout << "Spatially sparse CNN: input size " << inputSpatialSize; */
  // for (int i = 1; i < dimension; ++i)
    /* std::cout << "x" << inputSpatialSize; */
  /* std::cout << std::endl; */
}
void SparseConvNetCUDA::processBatch(SpatiallySparseBatch &batch,
                                     float learningRate, float momentum,
                                     std::ofstream &f, std::ofstream &g) {
  if (batch.type == RESCALEBATCH) {
    float scalingUnderneath = 1;
    for (int i = 0; i < layers.size(); i++) {
      batch.interfaces[i + 1].sub->reset();
      layers[i]->forwards(batch, batch.interfaces[i], batch.interfaces[i + 1]);
      /* std::cout << i << ":"
                << batch.interfaces[i].sub->features.size() * sizeof(float) /
                       (1 << 20) << "MB "; */
      layers[i]->scaleWeights(batch.interfaces[i], batch.interfaces[i + 1],
                              scalingUnderneath, i == layers.size() - 1);
    }
  } else {
    for (int i = 0; i < layers.size(); i++) {
      batch.interfaces[i + 1].sub->reset();
      layers[i]->forwards(batch, batch.interfaces[i], batch.interfaces[i + 1]);
    }
  }
  SoftmaxClassifier(batch.interfaces.back(), batch, nTop, memStream);
  if (batch.type == TRAINBATCH)
    for (int i = layers.size() - 1; i >= 0; i--) {
      layers[i]->backwards(batch, batch.interfaces[i], batch.interfaces[i + 1],
                           learningRate, momentum);
    }
  if (f)
    for (int j = 0; j < batch.predictions.size(); j++) {
      for (int k = 0; k < batch.predictions[j].size(); k++) {
        if (k > 0)
          f << " ";
        f << batch.predictions[j][k];
      }
      f << std::endl;
    }
  if (g)
    for (int j = 0; j < batch.predictions.size(); j++) {
      for (int k = 0; k < batch.probabilities[j].size(); k++) {
        if (k > 0)
          g << " ";
        g << batch.probabilities[j][k];
      }
      g << std::endl;
    }
}

activation SparseConvNetCUDA::processBatchForward(SpatiallySparseBatch &batch) {
    for (int i = 0; i < layers.size(); i++) {
      batch.interfaces[i + 1].sub->reset();
      layers[i]->forwards(batch, batch.interfaces[i], batch.interfaces[i + 1]);
    }

    SpatiallySparseBatchInterface &last_layer_batch_inteface = batch.interfaces.back();
    last_layer_batch_inteface.sub->features.copyToCPUAsync(memStream);
    const std::vector<float> features = last_layer_batch_inteface.sub->features.hVector();
    //std::copy(features.cbegin(), features.cend(), std::ostream_iterator(std::cout, ","));
      activation last_layer_activation;
      last_layer_activation.grid_size = last_layer_batch_inteface.grids[0].mp.size();
      last_layer_activation.feature_size = features.size();
      last_layer_activation.nSpatialSites = last_layer_batch_inteface.nSpatialSites;
      last_layer_activation.spatialSize = last_layer_batch_inteface.spatialSize;
      last_layer_activation.nFeatures = last_layer_batch_inteface.nFeatures;
      last_layer_activation.features = features;

      for (SparseGridMap::iterator it = last_layer_batch_inteface.grids[0].mp.begin();
          it != last_layer_batch_inteface.grids[0].mp.end(); ++it) {
        last_layer_activation.sparse_grid.push_back(std::make_pair(it->first, it->second));
      }

    last_layer_batch_inteface.sub->features.copyToGPUAsync(memStream);
    return last_layer_activation;
}

void SparseConvNetCUDA::processBatchBackward(SpatiallySparseBatch &batch,
                                     float learningRate, float momentum,
                                     const std::vector<float> &dfeatures) {

  SpatiallySparseBatchInterface &input = batch.interfaces.back();

  assert(batch.type == TRAINBATCH);
  assert(batch.batchSize == input.nSpatialSites);
  assert(input.nFeatures == input.featuresPresent.size());

  input.sub->dfeatures.copyToCPU();
  input.sub->dfeatures.resize(input.nSpatialSites *
                                input.featuresPresent.size());

  input.sub->dfeatures.hVector() = dfeatures;
  input.sub->dfeatures.copyToGPU();

  cudaCheckError();

  for (int i = layers.size() - 1; i >= 0; i--) {
    layers[i]->backwards(batch, batch.interfaces[i], batch.interfaces[i + 1],
                         learningRate, momentum);
  }
}

pd_report SparseConvNetCUDA::processDataset(SpatiallySparseDataset &dataset,
                                        int batchSize, float learningRate,
                                        float momentum) {
  assert(dataset.pictures.size() > 0);
  float errorRate = 0, nll = 0;
  multiplyAddCount = 0;
  auto start = std::chrono::system_clock::now();
  std::ofstream f, g;
  BatchProducer bp(*this, dataset, inputSpatialSize, batchSize);
  if (dataset.type == UNLABELEDBATCH) {
    f.open("unlabelledData.predictions");
    g.open("unlabelledData.probabilities");
  }

  while (SpatiallySparseBatch *batch = bp.nextBatch()) {
    processBatch(*batch, learningRate, momentum, f, g);
    errorRate += batch->mistakes * 1.0 / dataset.pictures.size();
    nll += batch->negativeLogLikelihood * 1.0 / dataset.pictures.size();
  }
  auto end = std::chrono::system_clock::now();
  auto diff =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  pd_report report;
  report.errorRate = errorRate;
  report.nll = nll;
  report.MegaMultiplyAdds_per_sample = roundf(multiplyAddCount / dataset.pictures.size() / 1000000);
  report.time = diff / 1000000000L;
  report.GigaMultiplyAdds_per_s = roundf(multiplyAddCount / diff);
  report.rate = roundf(dataset.pictures.size() * 1000000000.0f / diff);
  return report;
}

std::vector<std::vector<float>> SparseConvNetCUDA::predict(
        SpatiallySparseDataset &dataset) {
  assert(dataset.pictures.size() > 0);
  std::vector<std::vector<float>> result_matrix;
//  result_matrix.resize(dataset.pictures.size());
  int batchSize = 1;
  float learningRate, momentum;
  std::ofstream f, g;
  BatchProducer bp(*this, dataset, inputSpatialSize, batchSize);
  while (SpatiallySparseBatch *batch = bp.nextBatch()) {
    processBatch(*batch, learningRate, momentum, f, g);
    for (int i = 0; i < batch->batchSize; i++) {
      result_matrix.push_back(batch->probabilities[i]);
    }
  }
  return result_matrix;
}

std::vector<struct activation> SparseConvNetCUDA::layer_activations(
        SpatiallySparseDataset &dataset) {
  assert(dataset.pictures.size() == 1);
  int batchSize = 1;
  std::vector<struct activation> activations;
  BatchProducer bp(*this, dataset, inputSpatialSize, batchSize);
  while (SpatiallySparseBatch *batch = bp.nextBatch()) {
    processBatchForward(*batch);
    for (int i = 0; i < batch->interfaces.size(); ++i){
      batch->interfaces[i].sub->features.copyToCPUAsync(memStream);
      std::vector<float> &features = batch->interfaces[i].sub->features.hVector();
      activations.push_back(activation());
      activations[i].grid_size = batch->interfaces[i].grids[0].mp.size();
      activations[i].feature_size = features.size();
      activations[i].nSpatialSites = batch->interfaces[i].nSpatialSites;
      activations[i].spatialSize = batch->interfaces[i].spatialSize;
      activations[i].nFeatures = batch->interfaces[i].nFeatures;
      activations[i].features = features;
      for (SparseGridMap::iterator it = batch->interfaces[i].grids[0].mp.begin();
          it != batch->interfaces[i].grids[0].mp.end(); ++it) {
        activations[i].sparse_grid.push_back(std::make_pair(it->first, it->second));
      }
    }
  }
  return activations;
}

std::vector<struct pd_report> SparseConvNetCUDA::processDatasetRepeatTest(
    SpatiallySparseDataset &dataset, int batchSize, int nReps,
    std::string predictionsFilename, std::string confusionMatrixFilename) {
  assert(dataset.pictures.size() > 0);
  multiplyAddCount = 0;
  auto start = std::chrono::system_clock::now();
  std::vector<std::vector<int>> votes(dataset.pictures.size());
  std::vector<std::vector<float>> probs(dataset.pictures.size());
  for (int i = 0; i < dataset.pictures.size(); ++i) {
    votes[i].resize(dataset.nClasses);
    probs[i].resize(dataset.nClasses);
  }
  std::vector<struct pd_report> reports;
  for (int rep = 1; rep <= nReps; ++rep) {
    BatchProducer bp(*this, dataset, inputSpatialSize, batchSize);
    while (SpatiallySparseBatch *batch = bp.nextBatch()) {
      std::ofstream f, g;
      processBatch(*batch, 0, 0, f, g);
      for (int i = 0; i < batch->batchSize; ++i) {
        int ii = batch->sampleNumbers[i];
        votes[ii][batch->predictions[i][0]]++;
        for (int j = 0; j < dataset.nClasses; ++j)
          probs[ii][j] += batch->probabilities[i][j];
      }
    }
    int errors = dataset.pictures.size();
    float nll = 0;
    for (int i = 0; i < dataset.pictures.size(); ++i) {
      std::vector<int> predictions = vectorTopIndices(probs[i], nTop);
      for (int j = 0; j < nTop; j++)
        if (predictions[j] == dataset.pictures[i]->label)
          errors--;
      nll -= log(
          std::max(probs[i][dataset.pictures[i]->label] / rep, (float)1.0e-15));
    }

    if (!predictionsFilename.empty()) {
      // std::cout << predictionsFilename << std::endl;
      std::ofstream f(predictionsFilename.c_str());
      if (!dataset.header.empty())
        f << dataset.header << std::endl;
      for (int i = 0; i < dataset.pictures.size(); ++i) {
        f << dataset.pictures[i]->identify();
        if (dataset.type != UNLABELEDBATCH)
          f << "," << dataset.pictures[i]->label;
        for (int j = 0; j < dataset.nClasses; ++j)
          f << "," << probs[i][j] / rep;
        f << std::endl;
      }
    }
    if (!confusionMatrixFilename.empty()) {
      std::vector<float> cm(dataset.nClasses * dataset.nClasses);
      for (int i = 0; i < dataset.pictures.size(); ++i)
        for (int j = 0; j < dataset.nClasses; ++j)
          cm[dataset.pictures[i]->label * dataset.nClasses + j] +=
              probs[i][j] / rep;
      std::ofstream f(confusionMatrixFilename.c_str());
      for (int i = 0; i < dataset.nClasses; ++i) {
        for (int j = 0; j < dataset.nClasses; ++j) {
          f << cm[i * dataset.nClasses + j] << " ";
        }
        f << std::endl;
      }
    }
    auto end = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - start).count();
    // buffer.clear();
    reports.push_back(pd_report());
    reports[rep - 1].errorRate = 1.0 * errors / dataset.pictures.size();
    reports[rep - 1].nll = nll / dataset.pictures.size();
    reports[rep - 1].MegaMultiplyAdds_per_sample = roundf(multiplyAddCount / dataset.pictures.size() / 1000000);
    reports[rep - 1].time = diff / 1000000000L;
    reports[rep - 1].GigaMultiplyAdds_per_s = roundf(multiplyAddCount / diff);
    reports[rep - 1].rate = roundf(dataset.pictures.size() * 1000000000.0f / diff);
  }
  return reports;
}
void SparseConvNetCUDA::loadWeights(std::string baseName, int epoch,
                                    bool momentum, int firstNlayers) {
  std::string filename = std::string(baseName) + std::string("_epoch-") +
                         std::to_string(epoch) + std::string(".cnn");
  std::ifstream f;
  f.open(filename.c_str(), std::ios::out | std::ios::binary);
  if (f) {
    /* std::cout << "Loading network parameters from " << filename << std::endl; */
  } else {
    std::cout << "Cannot find " << filename << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < std::min((int)layers.size(), firstNlayers); i++)
    layers[i]->loadWeightsFromStream(f, momentum);
  if (inputNormalizingConstants.size() > 0)
    f.read((char *)&inputNormalizingConstants[0],
           sizeof(float) * inputNormalizingConstants.size());
  f.close();
}
void SparseConvNetCUDA::saveWeights(std::string baseName, int epoch,
                                    bool momentum) {
  std::string filename = std::string(baseName) + std::string("_epoch-") +
                         std::to_string(epoch) + std::string(".cnn");
  std::ofstream f;
  f.open(filename.c_str(), std::ios::binary);
  if (f) {
    for (int i = 0; i < layers.size(); i++)
      layers[i]->putWeightsToStream(f, momentum);
    if (inputNormalizingConstants.size() > 0)
      f.write((char *)&inputNormalizingConstants[0],
              sizeof(float) * inputNormalizingConstants.size());
    f.close();
  } else {
    // std::cout << "Cannot write " << filename << std::endl;
    exit(EXIT_FAILURE);
  }
}
void SparseConvNetCUDA::processIndexLearnerBatch(SpatiallySparseBatch &batch,
                                                 float learningRate,
                                                 float momentum,
                                                 std::ofstream &f) {
  int n = layers.size();
  for (int i = 0; i < n - 1; i++) // Stop 1 early (unless it is a training
                                  // batch)
    layers[i]->forwards(batch, batch.interfaces[i], batch.interfaces[i + 1]);
  if (f.is_open()) {
    assert(batch.interfaces[n - 1].nFeatures ==
           batch.interfaces[n - 1].featuresPresent.size());
    for (int i = 0; i < batch.batchSize; i++) {
      f << batch.sampleNumbers[i] << " " << batch.labels.hVector()[i];
      for (int j = 0; j < batch.interfaces[n - 1].nFeatures; j++)
        f << " "
          << batch.interfaces[n - 1]
                 .sub->features
                 .hVector()[i * batch.interfaces[n - 1].nFeatures + j];
      f << std::endl;
    }
  }
  if (batch.type == TRAINBATCH) {
    static_cast<IndexLearnerLayer *>(layers[n - 1])->indexLearnerIndices =
        batch.sampleNumbers;
    layers[n - 1]->forwards(batch, batch.interfaces[n - 1],
                            batch.interfaces[n]);
    IndexLearner(batch.interfaces[n], batch, nTop, memStream);
    for (int i = n - 1; i >= 0; i--)
      layers[i]->backwards(batch, batch.interfaces[i], batch.interfaces[i + 1],
                           learningRate, momentum);
  }
}
float SparseConvNetCUDA::processIndexLearnerDataset(
    SpatiallySparseDataset &dataset, int batchSize, float learningRate,
    float momentum) {
  assert(dataset.pictures.size() > 0);
  float errorRate = 0, nll = 0;
  auto start = std::chrono::system_clock::now();
  multiplyAddCount = 0;
  std::ofstream f;
  BatchProducer bp(*this, dataset, inputSpatialSize, batchSize);
  if (dataset.type != TRAINBATCH) {
    std::string filename = dataset.name + ".features";
    f.open(filename.c_str());
  }
  while (SpatiallySparseBatch *batch = bp.nextBatch()) {
    processIndexLearnerBatch(*batch, learningRate, momentum, f);
    errorRate += batch->mistakes * 1.0 / dataset.pictures.size();
    nll += batch->negativeLogLikelihood * 1.0 / dataset.pictures.size();
  }
  auto end = std::chrono::system_clock::now();
  auto diff =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  // if (dataset.type == TRAINBATCH)
    /* std::cout << dataset.name << " Mistakes:" << 100 * errorRate
              << "% NLL:" << nll << " MegaMultiplyAdds/sample:"
              << roundf(multiplyAddCount / dataset.pictures.size() / 1000000)
              << " time:" << diff / 1000000000L
              << "s GigaMultiplyAdds/s:" << roundf(multiplyAddCount / diff)
              << " rate:"
              << roundf(dataset.pictures.size() * 1000000000.0f / diff) << "/s"
              << std::endl; */
  return nll;
}
void SparseConvNetCUDA::processBatchDumpTopLevelFeaturess(
    SpatiallySparseBatch &batch, std::ofstream &f) { // editted: test
  int n = layers.size();
  for (int i = 0; i < layers.size() - 1; i++) {
    layers[i]->forwards(batch, batch.interfaces[i], batch.interfaces[i + 1]);
  }
  assert(batch.interfaces[n - 1].nFeatures ==
         batch.interfaces[n - 1].featuresPresent.size());
  for (int i = 0; i < batch.batchSize; i++) {
    f << batch.sampleNumbers[i] << " " << batch.labels.hVector()[i];
    for (int j = 0; j < batch.interfaces[n - 1].nFeatures; j++)
      f << " "
        << batch.interfaces[n - 1]
               .sub->features
               .hVector()[i * batch.interfaces[n - 1].nFeatures + j];
    f << std::endl;
  }
}
void SparseConvNetCUDA::processDatasetDumpTopLevelFeatures(
    SpatiallySparseDataset &dataset, int batchSize, int reps) {
  assert(dataset.pictures.size() > 0);
  std::ofstream f;
  assert(dataset.type != TRAINBATCH);
  std::string filename = dataset.name + ".features";
  f.open(filename.c_str());
  for (int i = 0; i < reps; i++) {
    BatchProducer bp(*this, dataset, inputSpatialSize, batchSize);
    while (SpatiallySparseBatch *batch = bp.nextBatch()) {
      processBatchDumpTopLevelFeaturess(*batch, f);
    }
  }
}

void SparseConvNetCUDA::calculateInputRegularizingConstants(
    SpatiallySparseDataset dataset) { // make copy of the dataset
  inputNormalizingConstants.resize(
      0); // Make sure input features rescaling is turned off.
  /* std::cout << "Using " << std::min(10000, (int)dataset.pictures.size())
            << " out of " << dataset.pictures.size()
            << " training samples to calculate regularizing constants."
            << std::endl; */
  if (dataset.pictures.size() > 10000)
    dataset.pictures.resize(10000);
  dataset.type = TESTBATCH; // pretend it is a test batch to turn off dropout
                            // and training data augmentation
  BatchProducer bp(*this, dataset, inputSpatialSize, 100);
  std::vector<float> c(nInputFeatures, 0);
  while (SpatiallySparseBatch *batch = bp.nextBatch()) {
    batch->interfaces[0].sub->features.copyToCPUAsync(memStream);
    std::vector<float> &features = batch->interfaces[0].sub->features.hVector();
    for (int i = 0; i < features.size(); ++i)
      c[i % nInputFeatures] =
          std::max(c[i % nInputFeatures], std::fabs(features[i]));
  }
  for (int i = 0; i < nInputFeatures; ++i) {
    inputNormalizingConstants.push_back(c[i] > 0 ? 1.0f / c[i] : 0);
    /* std::cout << inputNormalizingConstants.back() << " "; */
  }
  /* std::cout << std::endl; */
}
