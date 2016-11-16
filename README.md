PySparseConvNet
---------------

Fork of [Ben Graham's](http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/graham/)[SparseConvNet](https://github.com/btgraham/SparseConvNet) project but with a wrapper written in Cython. Optimized to process 3D mesh objects.


The SparseConvNet Library is written in C++ programming language, and utilizes a lot of CUDA capabilities for speed and efficiency.
But it is very limited when it comes to
* extending functionality — class structure and CUDA kernels are very complex, and requires re-compilation on every modification.
* changing loss functions — only learning configuration was SoftMax with log-likelihood loss function.
* Fine grained access to layer activations — there were no way to extract activations and therefore features from hidden layers.
* interactivity for exploration of models — every experiment was a compiled binary with now way to perform operations step by step,
to explore properties of models.

Because of all these problems we developed PySparseConvNet to solves them. On implementation level it’s a python compiled module that
can be used by Python interpreter, and harness all of it’s powerful features. Most of modern Deep Learning tools, such as (Theano, Chainer, Tensorflow) using Python as a way to perofrm interactive computing.


Interface of PySparseConvNet is much simpler, and consist’s of 4 classes:
* SparseNetwork — Network object class, it has all the methods to changing it’s structure, manipulate weights and activations.
* SparseDataset — Container class for sparse samples and their labels.
* SparseBatch — Gives access to data in dataset when brocessing separate mini-batches.
* Off3DPicture — Wrapper class for 3D models in OFF (Object File Format), used to voxelize samples to be processed by SparseNetwork.



**************************************************************************
PySparseConvNet is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SparseConvNet is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
[GNU General Public License](http://www.gnu.org/licenses/) for more details.
**************************************************************************
