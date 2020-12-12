Building and Installing
=======================

Building or installing from `vcpkg <https://github.com/microsoft/vcpkg>`_ is the simplest way to get started with flashlight. See the `top-level readme <https://github.com/facebookresearch/flashlight/blob/master/README.md>`_ for instructions on getting started with a ``vcpkg`` installation .For a more advanced installation from source, follow the steps below.

First, clone flashlight from `its repository on Github <https://github.com/facebookresearch/flashlight>`_:

::

   git clone https://github.com/facebookresearch/flashlight.git

Build Requirements
~~~~~~~~~~~~~~~~~~

- A C++ compiler with good C++14 support (e.g. g++ >= 5)
- `cmake <https://cmake.org/>`_ -- version 3.10 or later, and ``make``

Dependencies
------------

flashlight can be built with either a CUDA, or CPU (in development, will be shifted to OpenCL) backend. Requirements vary depending on which backend is selected.

- For all backends, `ArrayFire <https://github.com/arrayfire/arrayfire/wiki>`_ >= `3.7.1 <https://github.com/arrayfire/arrayfire/releases/tag/v3.7.1>`_ is required. flashlight can also be built flashlight with `ArrayFire 3.6.2 <https://github.com/arrayfire/arrayfire/releases/tag/v3.6.2>`_ - `3.6.4 <https://github.com/arrayfire/arrayfire/releases/tag/v3.6.4>`_, but only using commits ``<= 5518d91b7f4fd5b400cbc802cfbecc0df57836bd``.

  - Using ArrayFire >= 3.7.1 enables features that significantly improve performance; using it is highly recommended.

  - Using ArrayFire 3.7.2 is not recommended due to several bugs that are fixed in 3.7.3.

- The following dependencies are `downloaded, built, and installed automatically` with flashlight but can also be built and installed manually:

  - `Cereal <https://github.com/USCiLab/cereal>`_ is required for serialization -- the `develop` branch is used.

  - If building tests, `Google Test <https://github.com/google/googletest>`_ >= 1.10.0 is used.

  - If using CUDA <= 11, `NVIDIA CUB <https://github.com/NVlabs/cub>`_ is required.


Distributed Training Dependencies
---------------------------------
Building with distributed training is optional. See ``Build Options`` below.

- Regardless of backend, running with distributed training requires an MPI installation. `OpenMPI <https://www.open-mpi.org/>`_ is recommended if supported on your OS. On most Linux-based systems, ``sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev`` is sufficient.
- A BLAS implementation is required. If using `Intel MKL <https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html>`_, setting the ``MKLROOT`` environment variable can help CMake find libraries and headers. ``export MKLROOT=/opt/intel/mkl`` is sufficient on most Linux-based systems.
- If building the CUDA backend, `NVIDIA's NCCL library <https://developer.nvidia.com/nccl>`_ is required. Flashlight has been tested with NCCL 2.2.13.
- If building with the CPU backend, `Facebook's Gloo library <https://github.com/facebookincubator/gloo>`_ is required, and must be built with MPI; after installing OpenMPI, pass the ``-DUSE_MPI=ON`` flag to ``cmake`` when building Gloo.

CUDA Backend Dependencies
-------------------------

- CUDA >= 9.2 is required. flashlight has been tested with `CUDA 9.2 <https://developer.nvidia.com/cuda-92-download-archive>`_ and less extensively with CUDA 10.0.
- CUDNN >= 7.1.2 is required. flashlight has been tested with `CUDNN 7.1.2 <https://developer.nvidia.com/rdp/cudnn-archive>`_.

CPU Backend Dependencies
------------------------

The CPU backend is currently under active development. Required dependencies include:

- `oneDNN <https://github.com/oneapi-src/oneDNN>`_. oneDNN >= `v2.0 <https://github.com/oneapi-src/oneDNN/releases/tag/v2.0>`_ is required.

OpenCL Backend Dependencies
---------------------------

The OpenCL backend is currently under active development.

Build Instructions
~~~~~~~~~~~~~~~~~~
Build Options
-------------
+-------------------------+-------------------+---------------+
| Options                 | Configurations    | Default Value |
+=========================+===================+===============+
| FL_BACKEND              | CUDA, CPU, OPENCL | CUDA          |
+-------------------------+-------------------+---------------+
| FL_BUILD_CORE_ONLY      | ON, OFF           | OFF           |
+-------------------------+-------------------+---------------+
| FL_BUILD_LIBRARIES_ONLY | ON, OFF           | OFF           |
+-------------------------+-------------------+---------------+
| FL_BUILD_DISTRIBUTED    | ON, OFF           | ON            |
+-------------------------+-------------------+---------------+
| FL_BUILD_CONTRIB        | ON, OFF           | ON            |
+-------------------------+-------------------+---------------+
| FL_BUILD_EXPERIMENTAL   | ON, OFF           | OFF           |
+-------------------------+-------------------+---------------+
| FL_BUILD_TESTS          | ON, OFF           | ON            |
+-------------------------+-------------------+---------------+
| FL_BUILD_EXAMPLES       | ON, OFF           | ON            |
+-------------------------+-------------------+---------------+
| FL_BUILD_APP_ASR        | ON, OFF           | ON            |
+-------------------------+-------------------+---------------+
| FL_BUILD_APP_IMGCLASS   | ON, OFF           | ON            |
+-------------------------+-------------------+---------------+
| FL_BUILD_APP_LM         | ON, OFF           | ON            |
+-------------------------+-------------------+---------------+
| FL_BUILD_STANDALONE     | ON, OFF           | ON            |
+-------------------------+-------------------+---------------+
| CMAKE_BUILD_TYPE        | CMake build types | Debug         |
+-------------------------+-------------------+---------------+


Building on Linux or MacOS
--------------------------
flashlight has been thoroughly tested on Ubuntu 16.04 and above, CentOS 7.5, and macOS 10.14, but has good compatability with older operating systems.

Building from source on Linux and MacOS is simple:

.. code-block:: shell

  # in the flashlight project directory:
  mkdir -p build
  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DFLASHLIGHT_BACKEND=[backend] # valid backend
  make -j4  # (or any number of threads)
  make test

To change the location of the install, simply `set CMake's <https://cmake.org/cmake/help/v3.5/variable/CMAKE_INSTALL_PREFIX.html>`_ ``CMAKE_INSTALL_PREFIX`` before running ``cmake``, then:

.. code-block:: shell

 make install

To build a shared object, simply `set CMake's <https://cmake.org/cmake/help/v3.5/variable/BUILD_SHARED_LIBS.html>`_ ``BUILD_SHARED_LIBS`` when running ``cmake``.

Building on Windows
-------------------
Building flashlight on Windows is not supported at this time (coming soon).

Building/Running flashlight with Docker
---------------------------------------
flashlight and its dependencies can also be built with the provided Dockerfile.

To build flashlight with Docker:

- Install `Docker <https://docs.docker.com/engine/installation/>`_
- For CUDA backend install `nvidia-docker <https://github.com/NVIDIA/nvidia-docker/>`_
- Run the given Dockerfile in a new container:

.. code-block:: shell

 # for CUDA backend
 sudo docker run --runtime=nvidia --rm -itd --ipc=host --name flashlight flml/flashlight:cuda-latest
 # for CPU backend
 sudo docker run --rm -itd --name flashlight flml/flashlight:cpu-latest
 # go to terminal in the container
 sudo docker exec -it flashlight bash

- to run tests inside a container

.. code-block:: shell

 cd /root/flashlight/build && make test

- Build Docker image from source:

.. code-block:: shell

 git clone --recursive https://github.com/facebookresearch/flashlight.git
 cd flashlight
 # for CUDA backend
 sudo docker build -f ./Dockerfile-CUDA -t flashlight .
 # for CPU backend
 sudo docker build -f ./Dockerfile-CPU -t flashlight .

Building Your Project with flashlight
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The flashlight build exports the following CMake targets on install:

- ``flashlight::fl-libraries`` -- contains flashlight libraries headers and symbols.

- ``flashlight::flashlight`` -- contains flashlight libraries as well as the flashlight core autograd and neural network library.

- ``flashlight::flashlight-app-asr`` -- contains the automatic speech recognition application along with the flashlight core and flashlight libraries.

- ``flashlight::flashlight-app-imgclass`` -- contains the image classification application along with the flashlight core and flashlight libraries.

- ``flashlight::flashlight-app-lm`` -- contains the language modeling application along with the flashlight core and flashlight libraries.

Once flashlight is built and installed, including it in another project is simple using a CMake imported target. Suppose we have a project in ``project.cpp`` that uses flashlight:

::

   #include <iostream>

   #include <arrayfire.h>
   #include "flashlight/fl/flashlight.h"

   /**
    * ###### #         ##    ####  #    # #      #  ####  #    # #####
    * #      #        #  #  #      #    # #      # #    # #    #   #
    * #####  #       #    #  ####  ###### #      # #      ######   #
    * #      #       ######      # #    # #      # #  ### #    #   #
    * #      #       #    # #    # #    # #      # #    # #    #   #
    * #      ####### #    #  ####  #    # ###### #  ####  #    #   #
    */
   int main() {
     fl::Variable v(af::constant(1, 1), true);
     auto result = v + 10;
     std::cout << "Hello World!" << std::endl;
     af::print("Array value is ", result.array()); // 11.000
     return 0;
   }

We can link flashlight with the following CMake configuration:

.. code-block:: shell

  cmake_minimum_required(VERSION 3.10)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  find_package(flashlight CONFIG REQUIRED)

  # ...

  add_executable(myProject project.cpp)

  # the correct ArrayFire backend is transitively included by flashlight
  target_link_libraries(
    myProject
    PRIVATE
    # If building the package directly:
    flashlight::flashlight
  )

The above will automatically link all flashlight backend-specific dependencies and will add the correct directories to the target's (``myProject``'s) include directories.
