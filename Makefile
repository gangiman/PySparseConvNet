NVCC=nvcc
LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc -lcublas -larmadillo -lpython2.7
CYTHON_BUILD_DIR=build/temp.macosx-10.10-x86_64-2.7
ifeq ($(shell uname -s),Linux)
    LIBS += -lrt
    CYTHON_BUILD_DIR=build/temp.linux-x86_64-2.7
endif

clean:
	rm -r build/ *.pyc *.cpp *.so || true
fullclean: clean
	@$(MAKE) -C SparseConvNet $(MAKECMDGOALS)

build: clean
	python setup.py build_ext -if || true

test:
	python -m unittest tests/

full: build
	@$(MAKE) -C SparseConvNet $(MAKECMDGOALS)
	$(NVCC) --shared  -o PySparseConvNet.so $(CYTHON_BUILD_DIR)/sparseNetwork.o SparseConvNet/*.o $(LIBS)

update: build
	$(NVCC) --shared  -o PySparseConvNet.so $(CYTHON_BUILD_DIR)/sparseNetwork.o SparseConvNet/*.o $(LIBS)
