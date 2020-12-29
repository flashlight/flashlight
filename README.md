# Flashlight: Fast, Flexible Machine Learning in C++

[**Quickstart**](#quickstart)
| [**Installation**](#installation)
| [**Documentation**](https://fl.readthedocs.io/en/latest/)

[![CircleCI](https://circleci.com/gh/facebookresearch/flashlight.svg?style=shield)](https://circleci.com/gh/facebookresearch/flashlight)
[![Documentation Status](https://img.shields.io/readthedocs/fl.svg)](https://fl.readthedocs.io/en/latest/)
[![Docker Image Build Status](https://img.shields.io/github/workflow/status/facebookresearch/flashlight/Publish%20Docker%20images?label=docker%20image%20build)](https://hub.docker.com/r/flml/flashlight/tags)
[![Join the chat at https://gitter.im/flashlight-ml/community](https://img.shields.io/gitter/room/flashlight-ml/community)](https://gitter.im/flashlight-ml/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Flashlight is a fast, flexible machine learning library written entirely in C++
from the Facebook AI Research Speech team and the creators of Torch and
Deep Speech. Its core features include:
- Just-in-time kernel compilation with modern C++ with the [ArrayFire](https://github.com/arrayfire/arrayfire)
tensor library.
- CUDA, CPU, and OpenCL (coming soon) backends for GPU and CPU training.
- An emphasis on efficiency and scale.

Native support in C++ and simple extensibility makes Flashlight a powerful research framework that's *hackable to its core* and enables fast iteration on new experimental setups and algorithms without sacrificing performance. In a single repository, Flashlight provides [applications](https://github.com/facebookresearch/flashlight/tree/master/flashlight/app) for research across multiple domains:
- [Automatic speech recognition](https://github.com/facebookresearch/flashlight/tree/master/flashlight/app/asr) (the [wav2letter](https://github.com/facebookresearch/wav2letter/) project) — [Documentation](https://github.com/facebookresearch/flashlight/blob/tutorial_docs/flashlight/app/asr) | [Tutorial](https://github.com/facebookresearch/flashlight/blob/tutorial_docs/flashlight/app/asr/tutorial)
- [Image classification](https://github.com/facebookresearch/flashlight/tree/master/flashlight/app/imgclass)
- [Language modeling](https://github.com/facebookresearch/flashlight/tree/master/flashlight/app/lm)
- Image segmentation (coming soon)


### Project Layout

Flashlight is broken down into a few parts:
- [**`flashlight/lib`**](https://github.com/facebookresearch/flashlight/tree/master/flashlight/lib) contains kernels and standalone utilities for sequence losses, beam search decoding, text processing, and more.
- [**`flashlight/fl`**](https://github.com/facebookresearch/flashlight/tree/master/flashlight/fl) is the core neural network library using the [ArrayFire](https://github.com/arrayfire/arrayfire) tensor library.
- [**`flashlight/app`**](https://github.com/facebookresearch/flashlight/tree/master/flashlight/app) are applications of the core library to machine learning across domains.
- [**`flashlight/ext`**](https://github.com/facebookresearch/flashlight/tree/master/flashlight/ext) are extensions on top of Flashlight and ArrayFire that are useful across applications.

## Quickstart

First, [install Flashlight](#installation). And [link it to your own project](https://fl.readthedocs.io/en/latest/installation.html#building-your-project-with-flashlight).

[`Sequential`](https://fl.readthedocs.io/en/latest/modules.html#sequential) forms a sequence of Flashlight [`Module`](https://fl.readthedocs.io/en/latest/modules.html#module)s for chaining computation.

<details><summary>Implementing a simple convnet is easy.</summary>

```c++
#include <flashlight/fl/flashlight.h>

Sequential model;

model.add(View(af::dim4(IM_DIM, IM_DIM, 1, -1)));
model.add(Conv2D(
    1 /* input channels */,
    32 /* output channels */,
    5 /* kernel width */,
    5 /* kernel height */,
    1 /* stride x */,
    1 /* stride y */,
    PaddingMode::SAME; /* padding mode */,
    PaddingMode::SAME; /* padding mode */));
model.add(ReLU());
model.add(Pool2D(
    2 /* kernel width */,
    2 /* kernel height */,
    2 /* stride x */,
    2 /* stride y */));
model.add(Conv2D(32, 64, 5, 5, 1, 1, PaddingMode::SAME;, PaddingMode::SAME;));
model.add(ReLU());
model.add(Pool2D(2, 2, 2, 2));
model.add(View(af::dim4(7 * 7 * 64, -1)));
model.add(Linear(7 * 7 * 64, 1024));
model.add(ReLU());
model.add(Dropout(0.5));
model.add(Linear(1024, 10));
model.add(LogSoftmax());
```

Performing forward and backward computation is straightforwards:
```c++
auto output = model.forward(input);
auto loss = categoricalCrossEntropy(output, target);
loss.backward();
```

</details>

See the [MNIST example](https://fl.readthedocs.io/en/latest/mnist.html) for a full tutorial including a training loop and dataset abstractions.

[`Variable`](https://fl.readthedocs.io/en/latest/variable.html) is the base Fashlight tensor that operates on [ArrayFire `array`s](http://arrayfire.org/docs/classaf_1_1array.htm). Tape-based [Automatic differentiation in Flashlight](https://fl.readthedocs.io/en/latest/autograd.html) is simple and works as you'd expect.

<details><summary>Autograd Example</summary>

```c++
auto A = Variable(af::randu(1000, 1000), true /* calcGrad */);
auto B = 2.0 * A;
auto C = 1.0 + B;
auto D = log(C);
D.backward(); // populates A.grad() along with gradients for B, C, and D.
```

</details>

## Installation
### Requirements
At minimum, compilation requires:
- A C++ compiler with good C++14 support (e.g. gcc/g++ >= 5)
- [CMake](https://cmake.org/) -- version 3.10 or later, and ``make``
- A Unix-ish operating system. We're currently exploring experimental support on Windows.

#### Dependencies
<div class="tg-wrap"><table>
<thead>
  <tr>
    <th>Component</th>
    <th>Backend</th>
    <th>Dependencies</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">libraries</td>
    <td>CUDA</td>
    <td><a href="https://developer.nvidia.com/cuda-downloads">CUDA</a> &gt;= 9.2</td>
  </tr>
  <tr>
    <td>CPU</td>
    <td>A BLAS library (<a href="https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html">Intel MKL</a> &gt;= 2018, OpenBLAS, etc)</td>
  </tr>
  <tr>
    <td rowspan="3">core</td>
    <td>Any</td>
    <td><a href="https://github.com/arrayfire/arrayfire#installation">ArrayFire</a> &gt;= 3.7.3, an MPI library^(<a href="https://www.open-mpi.org/">OpenMPI</a>, etc),&nbsp;&nbsp;<a href="https://github.com/USCiLab/cereal">cereal</a>* &gt;= 1.3.0, <a href="https://github.com/nothings/stb">stb</a>*</td>
  </tr>
  <tr>
    <td>CUDA</td>
    <td><a href="https://developer.nvidia.com/cuda-downloads">CUDA</a> &gt;= 9.2, <a href="https://developer.nvidia.com/nccl">NCCL</a>^, <a href="https://developer.nvidia.com/cuDNN">cuDNN</a></td>
  </tr>
  <tr>
    <td>CPU</td>
    <td><a href="https://github.com/oneapi-src/oneDNN">oneDNN</a> &gt;= 2.0, <a href="https://github.com/facebookincubator/gloo">gloo</a> (<a href="https://github.com/facebookincubator/gloo/blob/01e2c2660cd43963ce1fe3e21220ac01f07d9a4b/docs/rendezvous.md#using-mpi">with MPI</a>)*^</td>
  </tr>
  <tr>
    <td>applications: all </td>
    <td>All</td>
    <td><a href="https://github.com/google/glog">Google Glog</a>, </td>
  </tr>
  <tr>
    <td>application: asr</td>
    <td>All</td>
    <td><a href="https://github.com/libsndfile/libsndfile">libsndfile</a>* &gt;= 10.0.28, a BLAS library (<a href="https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html">Intel MKL</a> &gt;= 2018, OpenBLAS, etc)</td>
  </tr>
  <tr>
    <td>application: imgclass</td>
    <td>All</td>
    <td>-</td>
  </tr>
  <tr>
    <td>application: lm</td>
    <td>All</td>
    <td>-</td>
  </tr>
  <tr>
    <td>tests</td>
    <td>All</td>
    <td><a href="https://github.com/google/googletest">Google Test (gtest, with gmock)</a>* &gt;= 1.10.0</td>
  </tr>
</tbody>
</table></div>

\* If not found on the system, this dependency is automatically downloaded and built from source.

^ Required if building with distributed training enabled. Distributed training is required for all applications.

### Building

There are two ways to work with Flashlight:
1. **As an installed library** that you link to with your own project.
2. **With in-source development** where you change the Flashlight project source and rebuild.

#### Installing as a Library

Flashlight is most-easily built and installed with `vcpkg`. Only the CUDA backend is currently supported with `vcpkg` (CPU backend coming soon). First, install [`CUDA` >= 9.2](https://developer.nvidia.com/cuda-downloads), [`cuDNN`](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html), and [`NCCL`](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html). Then, after [installing `vcpkg`](https://github.com/microsoft/vcpkg#getting-started) install the libraries and core with:
```shell
./vcpkg install flashlight-cuda
```
To see the features available for installation, run `./vcpkg search flashlight-cuda`. [Integrating Flashlight into your own project](https://vcpkg.readthedocs.io/en/latest/examples/installing-and-using-packages/#cmake) is simple. `vcpkg` [CMake toolchain integration](https://vcpkg.readthedocs.io/en/latest/examples/installing-and-using-packages/#cmake) is well-supported. OpenCL support in `vcpkg` is coming soon.

### In-Source Development

To build your clone of Flashlight from source using `vcpkg` and `CMake`, first install dependencies:
```shell
./vcpkg install \
    cuda intel-mkl fftw3 cub kenlm            \ # for flashlight libraries
    arrayfire[cuda] cudnn nccl openmpi cereal \ # for the flashlight neural net library
    gflags glog                               \ # for flashlight application libraries
    libsndfile                                \ # for the flashlight asr application
    stb                                       \ # for the flashlight imgclass application
    gtest                                       # optional, if building tests
```
Clone the repository:
```shell
git clone https://github.com/facebookresearch/flashlight.git && cd flashlight
mkdir -p build && cd build
```
Then, build from source using `vcpkg`'s CMake toolchain:
```shell
cmake .. \
    -DFL_BACKEND=CUDA
    -DCMAKE_TOOLCHAIN_FILE=[path to your vcpkg clone]/scripts/buildsystems/vcpkg.cmake
make -j$(nproc)
```
To build a subset of Flashlight's features, see the [installation options](https://fl.readthedocs.io/en/latest/installation.html) in the documentation.

#### Building from Source

<div class="tg-wrap"><table>
<thead>
  <tr>
    <th>Name</th>
    <th>Options</th>
    <th>Default Value</th>
    <th>Description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>FL_BACKEND</td>
    <td>CUDA, CPU, OPENCL</td>
    <td>CUDA</td>
    <td>Backend to build all components with.</td>
  </tr>
  <tr>
    <td>FL_BUILD_STANDALONE</td>
    <td>ON, OFF</td>
    <td>ON</td>
    <td>Downloads/builds some dependencies if not found.</td>
  </tr>
  <tr>
    <td>FL_BUILD_LIBRARIES</td>
    <td>ON, OFF</td>
    <td>ON</td>
    <td>Build the Flashlight libraries.</td>
  </tr>
  <tr>
    <td>FL_BUILD_CORE</td>
    <td>ON, OFF</td>
    <td>ON</td>
    <td>Build the Flashlight neural net library.</td>
  </tr>
  <tr>
    <td>FL_BUILD_DISTRIBUTED</td>
    <td>ON, OFF</td>
    <td>ON</td>
    <td>Build with distributed training; required for apps.</td>
  </tr>
  <tr>
    <td>FL_BUILD_CONTRIB</td>
    <td>ON, OFF</td>
    <td>ON</td>
    <td>Build contrib APIs subject to breaking changes.</td>
  </tr>
  <tr>
    <td>FL_BUILD_APPS</td>
    <td>ON, OFF</td>
    <td>ON</td>
    <td>Build applications (see below).</td>
  </tr>
  <tr>
    <td>FL_BUILD_APP_ASR</td>
    <td>ON, OFF</td>
    <td>ON</td>
    <td>Build the automatic speech recognition application.</td>
  </tr>
  <tr>
    <td>FL_BUILD_APP_IMGCLASS</td>
    <td>ON, OFF</td>
    <td>ON</td>
    <td>Build the image classification application.</td>
  </tr>
  <tr>
    <td>FL_BUILD_APP_LM</td>
    <td>ON, OFF</td>
    <td>ON</td>
    <td>Build the language modeling application.</td>
  </tr>
  <tr>
    <td>FL_BUILD_APP_ASR_TOOLS</td>
    <td>ON, OFF</td>
    <td>ON</td>
    <td>Build automatic speech recognition app tools.</td>
  </tr>
  <tr>
    <td>FL_BUILD_TESTS</td>
    <td>ON, OFF</td>
    <td>ON</td>
    <td>Build tests.</td>
  </tr>
  <tr>
    <td>FL_BUILD_EXAMPLES</td>
    <td>ON, OFF</td>
    <td>ON</td>
    <td>Build examples.</td>
  </tr>
  <tr>
    <td>FL_BUILD_EXPERIMENTAL</td>
    <td>ON, OFF</td>
    <td>OFF</td>
    <td>Build experimental components.</td>
  </tr>
  <tr>
    <td>CMAKE_BUILD_TYPE</td>
    <td>See <a href="https://cmake.org/cmake/help/v3.10/variable/CMAKE_BUILD_TYPE.html">docs</a>.</td>
    <td>Debug</td>
    <td>See the <a href="https://cmake.org/cmake/help/v3.10/variable/CMAKE_BUILD_TYPE.html">CMake documentation</a>.</td>
  </tr>
  <tr>
    <td>CMAKE_INSTALL_PREFIX</td>
    <td>[Directory]</td>
    <td>See <a href="https://cmake.org/cmake/help/v3.10/variable/CMAKE_INSTALL_PREFIX.html">docs</a>.</td>
    <td>See the <a href="https://cmake.org/cmake/help/v3.10/variable/CMAKE_INSTALL_PREFIX.html">CMake documentation</a>.</td>
  </tr>
</tbody>
</table></div>

Instructions to build fully from source can be found [in the documentation](https://fl.readthedocs.io/en/latest/installation.html).

### Contributing and Contact
Contact: vineelkpratap@fb.com, awni@fb.com, jacobkahn@fb.com, qiantong@fb.com, antares@fb.com, padentomasello@fb.com,
jcai@fb.com,  gab@fb.com, vitaliy888@fb.com, locronan@fb.com

Flashlight is being very actively developed. See
[CONTRIBUTING](CONTRIBUTING.md) for more on how to help out.

#### Acknowledgments
Some of Flashlight's code is derived from
[arrayfire-ml](https://github.com/arrayfire/arrayfire-ml/).

## License
Flashlight is under a BSD license. See [LICENSE](LICENSE) for more information.
