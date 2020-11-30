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

Native support in C++ and simple extensibility makes Flashlight a powerful research framework as  *hackable to its core* and enable fast iteration on new experimental setups and algorithms without sacrificing performance. In a single repository, Flashlight provides [applications](https://github.com/facebookresearch/flashlight/tree/master/flashlight/app) for research across multiple domains:
- [Automatic speech recognition](https://github.com/facebookresearch/flashlight/tree/master/flashlight/app/asr) (the [wav2letter](https://github.com/facebookresearch/wav2letter/) project)
- [Image classification](https://github.com/facebookresearch/flashlight/tree/master/flashlight/app/imclass)
- Language modeling (coming soon)
- Image segmentation (coming soon)


### Project Layout

Flashlight is broken down into a few parts:
- **`flashlight/lib`** contains kernels and standalone utilities for sequence losses, beam search decoding, text processing, and more.
- **`flashlight/fl`** is the core neural network library using the [ArrayFire](https://github.com/arrayfire/arrayfire) tensor library.
- **`flashlight/app`** are applications of the core library to machine learning across domains.
- **`flashlight/ext`** are extensions on top of Flashlight and ArrayFire that are useful across applications.

## Quickstart

First, [install Flashlight](#installation). And [link it to your own project](https://fl.readthedocs.io/en/latest/installation.html#building-your-project-with-flashlight).

[`Sequential`](https://fl.readthedocs.io/en/latest/modules.html#sequential) forms a sequence of Flashlight [`Module`](https://fl.readthedocs.io/en/latest/modules.html#module)s for chaining computation.

<details><summary>Implementing a simple convnet is easy...</summary>

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

### Building
Flashlight is most-easily built and installed with `vcpkg`. Only the CUDA backend is currently supported with `vcpkg`. First, install [`CUDA` >= 9.2](https://developer.nvidia.com/cuda-downloads), [`cuDNN`](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html), and [`NCCL`](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html). Then, after [installing `vcpkg`](https://github.com/microsoft/vcpkg#getting-started) install the libraries and core with:
```shell
./vcpkg install flashlight-cuda
```
To see the features available for installation, run `./vcpkg search Flashlight`. [Integrating Flashlight into your own project](https://vcpkg.readthedocs.io/en/latest/examples/installing-and-using-packages/#cmake) is simple. `vcpkg` [CMake toolchain integration](https://vcpkg.readthedocs.io/en/latest/examples/installing-and-using-packages/#cmake) is well-supported. OpenCL support in `vcpkg` is coming soon.

### In-Source Build

To build your clone of Flashlight from source using `vcpkg` and `CMake`, first install dependencies:
```shell
./vcpkg install \
    cuda intel-mkl fftw cub kenlm       \ # for flashlight libraries
    arrayfire cudnn nccl openmpi cereal \ # for the flashlight neural net library
    gflags                              \ # for flashlight application libraries
    libsndfile                          \ # for the flashlight asr application
    gtest                                 # optional, if building tests
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
