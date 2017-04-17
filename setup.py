# python neuralnet-setup.py build_ext --inplace

from distutils.core import setup
import numpy as np
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os


# Obtain the numpy include directory.
numpy_include = np.get_include()

base_sources = [
    "sparseNetwork.pyx",
    "_SparseConvNet.pxd"
]

sparse_conv_net_path = 'SparseConvNet/'


sparse_conv_net_sources = [
    # "BatchProducer.cu",
    # "ConvolutionalLayer.cu",
    # "ConvolutionalTriangularLayer.cu",
    # "IndexLearnerLayer.cu",
    # "MaxPoolingLayer.cu",
    # "MaxPoolingTriangularLayer.cu",
    # "NetworkArchitectures.cpp",
    # "NetworkInNetworkLayer.cu",
    # "NetworkInNetworkPReLULayer.cu",
    # "Picture.cpp",
    # "Regions.cu",
    # "Rng.cpp",
    # "SigmoidLayer.cu",
    # "SoftmaxClassifier.cu",
    # "SparseConvNet.cu",
    # "SparseConvNetCUDA.cu",
    # "SpatiallySparseBatch.cu",
    # "SpatiallySparseBatchInterface.cu",
    # "SpatiallySparseDataset.cpp",
    # "SpatiallySparseLayer.cu",
    # "TerminalPoolingLayer.cu",
    # "readImageToMat.cpp",
    # "types.cpp",
    # "utilities.cu",
    # "vectorCUDA.cu",
    # "ReallyConvolutionalLayer.cu",
    # "vectorHash.cpp"
]

gpp_flags = ["--std=c++11", "-fPIC", "-g" if os.environ.get("DEBUG", False) else "-O3"]

ext = Extension('PySparseConvNet',
                sources=base_sources + [
                    os.path.join(sparse_conv_net_path, _path)
                    for _path in sparse_conv_net_sources
                ],
                libraries=[
                    "opencv_core",
                    "opencv_highgui",
                    "opencv_imgproc",
                    "cublas",
                    "armadillo",
                    "python2.7"],
                language='c++',
                extra_compile_args={
                    'g++': gpp_flags,
                    'nvcc': ['-arch=sm_50', '--std=c++11', '-O3',
                             "-Xcompiler", "-fPIC"]
                },
                include_dirs=[numpy_include, '/usr/local/cuda/include'])


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kind of like a weird functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile
    self.linker_so = ["nvcc", "--shared",
                      # "-Xlinker", "--unresolved-symbols=ignore-all"
                      ]
    self.compiler_cxx = None

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', 'nvcc')
            postargs = extra_postargs['nvcc']
        elif os.path.splitext(src)[1] == '.cpp':
            self.set_executable('compiler_so', "g++")
            postargs = extra_postargs['g++']
        else:
            raise Exception("Unknown file extension.")

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

setup(
    name='PySparseConvNet',
    author='Alexandr Notchenko',
    version='0.1',
    ext_modules=[ext],
    # inject our custom trigger
    cmdclass={'build_ext': custom_build_ext})
