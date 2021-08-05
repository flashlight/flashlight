![Flashlight: Fast, Flexible Machine Learning in C++](./logo.svg)

<hr/>

[**Quickstart**](#quickstart)
| [**Installation**](#building-and-installing)
| [**Documentation**](https://fl.readthedocs.io/en/latest/)

[![CircleCI](https://circleci.com/gh/flashlight/flashlight.svg?style=shield)](https://app.circleci.com/pipelines/github/flashlight/flashlight)
[![Documentation Status](https://img.shields.io/readthedocs/fl.svg)](https://fl.readthedocs.io/en/latest/)
[![Docker Image Build Status](https://img.shields.io/github/workflow/status/flashlight/flashlight/Publish%20Docker%20images?label=docker%20image%20build)](https://hub.docker.com/r/flml/flashlight/tags)
[![Join the chat at https://gitter.im/flashlight-ml/community](https://img.shields.io/gitter/room/flashlight-ml/community)](https://gitter.im/flashlight-ml/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![codecov](https://codecov.io/gh/flashlight/flashlight/branch/master/graph/badge.svg?token=rBp4AilMc0)](https://codecov.io/gh/flashlight/flashlight)

[![Docker Image for CUDA backend](https://img.shields.io/docker/image-size/flml/flashlight/cuda-latest?label=docker%20%28cuda%29&logo=docker)](https://hub.docker.com/r/flml/flashlight/tags?page=1&ordering=last_updated&name=cuda-latest)
[![Docker Image for CPU backend](https://img.shields.io/docker/image-size/flml/flashlight/cpu-latest?label=docker%20%28cpu%29&logo=docker)](https://hub.docker.com/r/flml/flashlight/tags?page=1&ordering=last_updated&name=cpu-latest)

[![Install CUDA backend with vcpkg](https://img.shields.io/badge/dynamic/json?color=orange&label=get%20%28cuda%29&query=name&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmicrosoft%2Fvcpkg%2Fmaster%2Fports%2Fflashlight-cuda%2Fvcpkg.json&prefix=vcpkg%20install%20)](https://vcpkg.info/port/flashlight-cuda)
[![Install CPU backend with vcpkg](https://img.shields.io/badge/dynamic/json?color=orange&label=get%20%28cpu%29&query=name&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmicrosoft%2Fvcpkg%2Fmaster%2Fports%2Fflashlight-cpu%2Fvcpkg.json&prefix=vcpkg%20install%20)](https://vcpkg.info/port/flashlight-cpu)


Flashlight is a fast, flexible machine learning library written entirely in C++
from the Facebook AI Research Speech team and the creators of Torch and
Deep Speech. Its core features include:
- Just-in-time kernel compilation with modern C++ with the [ArrayFire](https://github.com/arrayfire/arrayfire)
tensor library.
- CUDA and CPU backends for GPU and CPU training.
- An emphasis on efficiency and scale.

Native support in C++ and simple extensibility makes Flashlight a powerful research framework that's *hackable to its core* and enables fast iteration on new experimental setups and algorithms without sacrificing performance. In a single repository, Flashlight provides [apps](https://github.com/flashlight/flashlight/tree/master/flashlight/app) for research across multiple domains:
- [Automatic speech recognition](https://github.com/flashlight/flashlight/tree/master/flashlight/app/asr) (the [wav2letter](https://github.com/flashlight/wav2letter/) project) — [Documentation](flashlight/app/asr) | [Tutorial](flashlight/app/asr/tutorial)
- [Image classification](flashlight/app/imgclass)
- [Object detection](flashlight/app/objdet)
- [Language modeling](flashlight/app/lm)


### Project Layout

Flashlight is broken down into a few parts:
- [**`flashlight/lib`**](flashlight/lib) contains kernels and standalone utilities for sequence losses, beam search decoding, text processing, and more.
- [**`flashlight/fl`**](flashlight/fl) is the core neural network library using the [ArrayFire](https://github.com/arrayfire/arrayfire) tensor library.
- [**`flashlight/app`**](flashlight/app) are applications of the core library to machine learning across domains.
- [**`flashlight/ext`**](flashlight/ext) are extensions on top of Flashlight and ArrayFire that are useful across apps.

## Quickstart

First, [build and install Flashlight](#building-and-installing) and [link it to your own project](#building-your-own-project-with-flashlight).

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

[`Variable`](https://fl.readthedocs.io/en/latest/variable.html) is the base Flashlight tensor that operates on [ArrayFire `array`s](http://arrayfire.org/docs/classaf_1_1array.htm). Tape-based [Automatic differentiation in Flashlight](https://fl.readthedocs.io/en/latest/autograd.html) is simple and works as you'd expect.

<details><summary>Autograd Example</summary>

```c++
auto A = Variable(af::randu(1000, 1000), true /* calcGrad */);
auto B = 2.0 * A;
auto C = 1.0 + B;
auto D = log(C);
D.backward(); // populates A.grad() along with gradients for B, C, and D.
```

</details>

## Building and Installing
[**Install with `vcpkg`**](#library-installation-with-vcpkg) | [**With Docker**](#building-and-running-flashlight-with-docker) | [**From Source**](#building-from-source) | [**From Source with `vcpkg`**](#from-source-build-with-vcpkg) | [**Build Your Project with Flashlight**](#building-your-own-project-with-flashlight)

### Requirements
At minimum, compilation requires:
- A C++ compiler with good C++17 support (e.g. gcc/g++ >= 7)
- [CMake](https://cmake.org/) — version 3.10 or later, and ``make``
- A Linux-based operating system.

See the [full dependency](#dependencies) list for more details if [building from source](#building-from-source).

Instructions for building/installing Python bindings [can be found here](bindings/python/README.md).

### Flashlight Build Setups

Flashlight can be broken down into several components as [described above](#project-layout). Each component can be incrementally built by specifying the correct [build options](#build-options).

There are two ways to work with Flashlight:
1. **As an installed library** that you link to with your own project. This is best for building standalone applications dependent on Flashlight.
2. **With in-source development** where the Flashlight project source is changed and rebuilt. This is best if customizing/hacking the core framework or the Flashlight-provided [app binaries](flashlight/app).

Flashlight can be built in one of two ways:
1. [**With `vcpkg`**](#installing-flashlight-with-vcpkg), a [C++ package manager](https://github.com/microsoft/vcpkg).
2. [**From source**](#building-from-source) by installing dependencies as needed.

### Installing Flashlight with `vcpkg`
#### Library Installation with `vcpkg`

Flashlight is most-easily built and installed with `vcpkg`. Both the CUDA and CPU backends are supported with `vcpkg`. For either backend, first install [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html). For the CUDA backend, install [`CUDA` >= 9.2](https://developer.nvidia.com/cuda-downloads), [`cuDNN`](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html), and [`NCCL`](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html). Then, after [installing `vcpkg`](https://github.com/microsoft/vcpkg#getting-started), install the libraries and core with:
```shell
./vcpkg/vcpkg install flashlight-cuda # CUDA backend, OR
./vcpkg/vcpkg install flashlight-cpu  # CPU backend
```
To install [Flashlight apps](flashlight/app), check the features available for installation by running `./vcpkg search flashlight-cuda` or `./vcpkg search flashlight-cpu`. Each app is a "feature": for example, `./vcpkg install flashlight-cuda[asr]` installs the ASR app with the CUDA backend.

Below is the currently-supported list of features (for each of [`flashlight-cuda`](https://vcpkg.info/port/flashlight-cuda) and [`flashlight-cpu`](https://vcpkg.info/port/flashlight-cpu)):
```
flashlight-{cuda/cpu}[lib]      # Flashlight libraries
flashlight-{cuda/cpu}[nn]       # Flashlight neural net library
flashlight-{cuda/cpu}[asr]      # Flashlight speech recognition app
flashlight-{cuda/cpu}[lm]       # Flashlight language modeling app
flashlight-{cuda/cpu}[imgclass] # Flashlight image classification app
```

Flashlight [app binaries](flashlight/app) are also built for the selected features and are installed into the `vcpkg` install tree's `tools` directory.

[Integrating Flashlight into your own project](#with-a-vcpkg-flashlight-installation) with is simple using `vcpkg`'s [CMake toolchain integration](https://vcpkg.readthedocs.io/en/latest/examples/installing-and-using-packages/#cmake).

#### From-Source Build with `vcpkg`

First, install the dependencies for your backend of choice using `vcpkg` (click to expand the below):

<details><summary>Installing CUDA Backend Dependencies with vcpkg</summary>

To build the Flashlight CUDA backend from source using dependencies installed with `vcpkg`, install [`CUDA` >= 9.2](https://developer.nvidia.com/cuda-downloads), [`cuDNN`](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html), [`NCCL`](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html), and [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html), then build the rest of the dependencies for the CUDA backend based on which Flashlight features you'd like to build:
```shell
./vcpkg install \
    cuda intel-mkl fftw3 cub kenlm                \ # if building flashlight libraries
    arrayfire[cuda] cudnn nccl openmpi cereal stb \ # if building the flashlight neural net library
    gflags glog                                   \ # if building any flashlight apps
    libsndfile                                    \ # if building the flashlight asr app
    gtest                                           # optional, if building tests
```
</details>

<details><summary>Installing CPU Backend Dependencies with vcpkg</summary>

To build the Flashlight CPU backend from source using dependencies installed with `vcpkg`, install [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html), then build the rest of the dependencies for the CPU backend based on which Flashlight features you'd like to build:
```shell
./vcpkg install \
    intel-mkl fftw3 kenlm                              \ # for flashlight libraries
    arrayfire[cpu] gloo[mpi] openmpi onednn cereal stb \ # for the flashlight neural net library
    gflags glog                                        \ # for any flashlight apps
    libsndfile                                         \ # for the flashlight asr app
    gtest                                                # optional, for tests
```

</details>

##### Build Using the `vcpkg` Toolchain File
To build Flashlight from source with these dependencies, clone the repository:
```shell
git clone https://github.com/flashlight/flashlight.git && cd flashlight
mkdir -p build && cd build
```
Then, build from source using `vcpkg`'s [CMake toolchain](https://github.com/microsoft/vcpkg/blob/master/docs/users/integration.md#cmake-toolchain-file-recommended-for-open-source-cmake-projects):
```shell
cmake .. \
    -DCMAKE_BUILD_TYPE=Release
    -DFL_BACKEND=CUDA
    -DCMAKE_TOOLCHAIN_FILE=[path to your vcpkg clone]/scripts/buildsystems/vcpkg.cmake
make -j$(nproc)
make install -j$(nproc) # only if you want to install Flashlight for external use
```
To build a subset of Flashlight's features, see the [build options](#build-options) below.

### Building from Source
To build from source, first install the below [dependencies](#dependencies). Most are available with your system's local package manager.

Some dependencies marked below are downloaded and installed automatically if not found on the local system. `FL_BUILD_STANDALONE` determines this behavior — if disabled, dependencies won't be downloaded and built when building Flashlight.

**Once all dependencies are installed**, clone the repository:
```shell
git clone https://github.com/flashlight/flashlight.git && cd flashlight
mkdir -p build && cd build
```
Then build all Flashlight components with:
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DFL_BACKEND=[backend] [...build options]
make -j$(nproc)
make install
```
Setting the `MKLROOT` environment variable (`export MKLROOT=/opt/intel/oneapi/mkl/latest` or `export MKLROOT=/opt/intel/mkl` on most Linux-based systems) can help CMake find Intel MKL if not initially found.

To build a smaller subset of Flashlight features/apps, see the [build options](#build-options) below for a complete list of options.

To install Flashlight in a custom directory, use CMake's [`CMAKE_INSTALL_PREFIX`](https://cmake.org/cmake/help/v3.10/variable/CMAKE_INSTALL_PREFIX.html) argument. Flashlight libraries can be built as shared libraries using CMake's [`BUILD_SHARED_LIBS`](https://cmake.org/cmake/help/v3.10/variable/BUILD_SHARED_LIBS.html) argument.

Flashlight uses modern CMake and `IMPORTED` targets for most dependencies. If a dependency isn't found, passing `-D<package>_DIR` to your `cmake` command or exporting `<package>_DIR` as an environment variable equal to the path to `<package>Config.cmake` can help locate dependencies on your system. See [the documentation](https://cmake.org/cmake/help/v3.10/command/find_package.html) for more details. If CMake is failing to locate a package, check to see if a corresponding [issue](https://github.com/flashlight/flashlight/issues) has already been created before creating your own.

#### Dependencies

Dependencies marked with `*` are automatically downloaded and built from source if not found on the system. Setting `FL_BUILD_STANDALONE` to `OFF` disables this behavior.

Dependencies marked with `^` are required if building with distributed training enabled (`FL_BUILD_DISTRIBUTED` — see the [build options](#build-options) below). Distributed training is required for all apps.

Dependencies marked with `†` are installable via `vcpkg`. See the [instructions for installing those dependencies](#from-source-build-with-vcpkg) above for doing a Flashlight from-source build.

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
    <td><a href="https://developer.nvidia.com/cuda-downloads">CUDA</a> &gt;= 9.2, <a href="https://github.com/nvidia/cub">CUB</a>*† (if CUDA &lt; 11)</td>
  </tr>
  <tr>
    <td>CPU</td>
    <td>A BLAS library (<a href="https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html">Intel MKL</a> &gt;= 2018, OpenBLAS†, etc)</td>
  </tr>
  <tr>
    <td rowspan="3">core</td>
    <td>Any</td>
    <td><a href="https://github.com/arrayfire/arrayfire#installation">ArrayFire</a> &gt;= 3.7.3†, an MPI library^(<a href="https://www.open-mpi.org/">OpenMPI</a>†, etc),&nbsp;&nbsp;<a href="https://github.com/USCiLab/cereal">cereal</a>*† &gt;= 1.3.0, <a href="https://github.com/nothings/stb">stb</a>*†</td>
  </tr>
  <tr>
    <td>CUDA</td>
    <td><a href="https://developer.nvidia.com/cuda-downloads">CUDA</a> &gt;= 9.2, <a href="https://developer.nvidia.com/nccl">NCCL</a>^, <a href="https://developer.nvidia.com/cuDNN">cuDNN</a></td>
  </tr>
  <tr>
    <td>CPU</td>
    <td><a href="https://github.com/oneapi-src/oneDNN">oneDNN</a>† &gt;= 2.0, <a href="https://github.com/facebookincubator/gloo">gloo</a> (<a href="https://github.com/facebookincubator/gloo/blob/01e2c2660cd43963ce1fe3e21220ac01f07d9a4b/docs/rendezvous.md#using-mpi">with MPI</a>)*^†</td>
  </tr>
  <tr>
    <td>app: all </td>
    <td>Any</td>
    <td><a href="https://github.com/google/glog">Google Glog</a>†, <a href="https://github.com/gflags/gflags">Gflags</a>†</td>
  </tr>
  <tr>
    <td>app: asr</td>
    <td>Any</td>
    <td><a href="https://github.com/libsndfile/libsndfile">libsndfile</a>*† &gt;= 10.0.28, a BLAS library (<a href="https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html">Intel MKL</a> &gt;= 2018, OpenBLAS†, etc)</td>
  </tr>
  <tr>
    <td>app: imgclass</td>
    <td>Any</td>
    <td>-</td>
  </tr>
  <tr>
    <td>app: objdet</td>
    <td>Any</td>
    <td>-</td>
  </tr>
  <tr>
    <td>app: lm</td>
    <td>Any</td>
    <td>-</td>
  </tr>
  <tr>
    <td>tests</td>
    <td>Any</td>
    <td><a href="https://github.com/google/googletest">Google Test (gtest, with gmock)</a>*† &gt;= 1.10.0</td>
  </tr>
</tbody>
</table></div>

#### Build Options
The Flashlight CMake build accepts the following build options (prefixed with `-D` when running CMake from the command line):

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
    <td>Backend with which to build all components.</td>
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
    <td>FL_BUILD_ALL_APPS</td>
    <td>ON, OFF</td>
    <td>OFF</td>
    <td>Defines default value for every app (see below).</td>
  </tr>
  <tr>
    <td>FL_BUILD_APP_ASR</td>
    <td>ON, OFF</td>
    <td>FL_BUILD_ALL_APPS</td>
    <td>Build the automatic speech recognition app.</td>
  </tr>
  <tr>
    <td>FL_BUILD_APP_IMGCLASS</td>
    <td>ON, OFF</td>
    <td>FL_BUILD_ALL_APPS</td>
    <td>Build the image classification app.</td>
  </tr>
    <tr>
    <td>FL_BUILD_APP_OBJDET</td>
    <td>ON, OFF</td>
    <td>FL_BUILD_ALL_APPS</td>
    <td>Build automatic speech recognition app tools.</td>
  </tr>
  <tr>
    <td>FL_BUILD_APP_LM</td>
    <td>ON, OFF</td>
    <td>FL_BUILD_ALL_APPS</td>
    <td>Build the language modeling app.</td>
  </tr>
  <tr>
    <td>FL_BUILD_APP_ASR_TOOLS</td>
    <td>ON, OFF</td>
    <td>FL_BUILD_APP_ASR</td>
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

### Building Your Own Project with Flashlight
Flashlight is most-easily linked to using CMake. Flashlight exports the following CMake targets when installed:
- `flashlight::fl-libraries` — contains flashlight libraries headers and symbols.
- `flashlight::flashlight` — contains flashlight libraries as well as the flashlight core autograd and neural network library.
- `flashlight::flashlight-app-asr` — contains the automatic speech recognition app along with the flashlight core and flashlight libraries.
- `flashlight::flashlight-app-imgclass` — contains the image classification app along with the flashlight core and flashlight libraries.
- `flashlight::flashlight-app-objdet` — contains the object detection app along with the flashlight core and flashlight libraries.
- `flashlight::flashlight-app-lm` — contains the language modeling app along with the flashlight core and flashlight libraries.

Given a simple `project.cpp` file that includes and links to Flashlight:
```c++
#include <iostream>

#include <arrayfire.h>
#include <flashlight/fl/flashlight.h>

int main() {
 fl::Variable v(af::constant(1, 1), true);
 auto result = v + 10;
 std::cout << "Hello World!" << std::endl;
 af::print("Array value is ", result.array()); // 11.000
 return 0;
}
```

The following CMake configuration links Flashlight and sets include directories:

```cmake
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(myProject project.cpp)

find_package(flashlight CONFIG REQUIRED)
target_link_libraries(myProject PRIVATE flashlight::flashlight)
```

#### With a `vcpkg` Flashlight Installation

If you installed Flashlight with `vcpkg`, the above CMake configuration for `myProject` can be built by running:
```shell
cd project && mkdir build && cd build
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg clone]/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

#### With a From-Source Flashlight Installation

If using a from-source installation of Flashlight, Flashlight will be found automatically by CMake:
```shell
cd project && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
If Flashlight is installed in a custom location using a `CMAKE_INSTALL_PREFIX`, passing `-Dflashlight_DIR=[install prefix]/share/flashlight/cmake` as an argument to your `cmake` command can help CMake find Flashlight.

### Building and Running Flashlight with Docker
Flashlight and its dependencies can also be built with the provided Dockerfiles — see the accompanying [Docker documentation](.docker) for more information.

### Contributing and Contact
Contact: vineelkpratap@fb.com, awni@fb.com, jacobkahn@fb.com, qiantong@fb.com, antares@fb.com, padentomasello@fb.com,
jcai@fb.com,  gab@fb.com, vitaliy888@fb.com, locronan@fb.com

Flashlight is being very actively developed. See
[CONTRIBUTING](CONTRIBUTING.md) for more on how to help out.

#### Acknowledgments
Some of Flashlight's code is derived from
[arrayfire-ml](https://github.com/arrayfire/arrayfire-ml/).

## License
Flashlight is under an MIT license. See [LICENSE](LICENSE) for more information.
