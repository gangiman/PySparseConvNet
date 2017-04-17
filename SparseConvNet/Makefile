CC=g++
DEBUG ?= 0
ifeq ($(DEBUG), 1)
    CFLAGS=--std=c++11 -g -fPIC
    NVCCFLAGS=--std=c++11 -arch sm_50 -g -Xcompiler -fPIC
else
    CFLAGS=--std=c++11 -O3 -fPIC
    NVCCFLAGS=--std=c++11 -arch sm_50 -O3 -Xcompiler -fPIC
endif
NVCC=nvcc
OBJ=BatchProducer.o ConvolutionalLayer.o ConvolutionalTriangularLayer.o IndexLearnerLayer.o MaxPoolingLayer.o MaxPoolingTriangularLayer.o NetworkInNetworkLayer.o NetworkInNetworkPReLULayer.o Picture.o Regions.o Rng.o SigmoidLayer.o SoftmaxClassifier.o SparseConvNet.o SparseConvNetCUDA.o SpatiallySparseBatch.o SpatiallySparseBatchInterface.o SpatiallySparseDataset.o SpatiallySparseLayer.o TerminalPoolingLayer.o types.o utilities.o vectorCUDA.o ReallyConvolutionalLayer.o vectorHash.o Off3DFormatPicture.o VoxelPicture.o
LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc -lcublas -larmadillo
ifeq ($(shell uname -s),Linux)
    LIBS += -lrt
endif

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(NVCCFLAGS)

clean:
	rm *.o

shrec2015: $(OBJ) Off3DFormatPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015.o
	$(NVCC) -o shrec2015 $(OBJ) Off3DFormatPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015.o $(LIBS) $(NVCCFLAGS)

modelnet: $(OBJ) Off3DFormatPicture.o SpatiallySparseDatasetModelNet.o modelnet.o
	$(NVCC) -o modelnet $(OBJ) Off3DFormatPicture.o SpatiallySparseDatasetModelNet.o modelnet.o $(LIBS) $(NVCCFLAGS)

shrec2015triangular: $(OBJ) Off3DFormatTriangularPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015triangular.o
	$(NVCC) -o shrec2015triangular $(OBJ) Off3DFormatTriangularPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015triangular.o $(LIBS) $(NVCCFLAGS)

full: $(OBJ)
fullclean: clean
