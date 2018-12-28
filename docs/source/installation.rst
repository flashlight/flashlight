Building and Installing
=======================
Currently, flashlight must be built and installed from source.

The installation creates ``libflashlight``, which contains the entire flashlight library. Headers are contained in ``flashlight/``, which is placed in the specified ``include`` directory after install.

First, clone flashlight from `its repository on Github <https://github.com/facebookresearch/flashlight>`_:

::

   git clone --recursive https://github.com/facebookresearch/flashlight.git


Build Requirements
~~~~~~~~~~~~~~~~~~

- A C++ compiler with good C++ 11 support (e.g. g++ >= 4.8)
- `cmake <https://cmake.org/>`_ -- version 3.5.1 or later, and ``make``

Dependencies
------------

flashlight can be built with either a CUDA, CPU (coming soon), or OpenCL (coming soon) backend. Requirements vary depending on which backend is selected.

- For all backends, `ArrayFire <https://github.com/arrayfire/arrayfire/wiki>`_ >= 3.6.1 is required. flashlight has been tested with `ArrayFire 3.6.1 <https://github.com/arrayfire/arrayfire/releases/tag/v3.6.1>`_.
- The following dependencies are `downloaded, built, and installed automatically` with flashlight:

  - `Cereal <https://github.com/USCiLab/cereal>`_ is required for serialization -- the `develop` branch must be used.
  - If building tests, `Google Test <https://github.com/google/googletest>`_ >= 1.8.0 is required.
    

Distributed Training Dependencies
---------------------------------
Building with distributed training is optional. See ``Build Options`` below.

- Regardless of backend, running with distributed training requires an MPI installation. `OpenMPI <https://www.open-mpi.org/>`_ is recommended if supported on your OS.
- If building the CUDA backend, `NVIDIA's NCCL library <https://developer.nvidia.com/nccl>`_ is required. Flashlight has been tested with NCCL >= 2.2.13.
- If building with the CPU backend, `Facebook's Gloo library <https://github.com/facebookincubator/gloo>`_ is required, and must be built with MPI.

CUDA Backend Dependencies
-------------------------

- CUDA >= 9.2 is required. Using `CUDA 9.2 <https://developer.nvidia.com/cuda-92-download-archive>`_ is recommended.
- CUDNN >= 7.2.1 is required. Using `CUDNN 7.2.1 <https://developer.nvidia.com/rdp/cudnn-archive>`_ is recommended.

CPU Backend Dependencies
------------------------

The CPU backend is not currently supported.

OpenCL Backend Dependencies
---------------------------

The OpenCL backend is not currently supported.

Build Instructions
~~~~~~~~~~~~~~~~~~
Build Options
-------------
+-------------------------+-------------------+---------------+
| Options                 | Configurations    | Default Value |
+=========================+===================+===============+
| FLASHLIGHT_BACKEND      | CUDA, CPU, OPENCL | CUDA          |
+-------------------------+-------------------+---------------+
| FL_BUILD_DISTRIBUTED    | ON, OFF           | ON            |
+-------------------------+-------------------+---------------+
| FL_BUILD_TESTS          | ON, OFF           | ON            |
+-------------------------+-------------------+---------------+
| FL_BUILD_EXAMPLES       | ON, OFF           | ON            |
+-------------------------+-------------------+---------------+
| CMAKE_BUILD_TYPE        | CMake build types | Debug         |
+-------------------------+-------------------+---------------+


Building on Linux
-----------------
flashlight has been thoroughly tested on Ubuntu 16.04 and above, CentOS 7.5, and macOS 10.14, but has good compatability with older operating systems.

Building on Linux is simple:

.. code-block:: shell

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

Building flashlight with Docker
------------
 - Install `Docker <https://docs.docker.com/engine/installation/>`_  and `nvidia-docker <https://github.com/NVIDIA/nvidia-docker/>`_
 - run container

 ::
 
     docker run --runtime=nvidia --rm -itd --ipc=host --name flashlight facebookreasearch/flashlight:cuda000
     docker exec -it flashlight bash


Building Your Project with flashlight
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once flashlight is built and installed, including it in another project is simple, using CMake. Suppose we have a project in ``project.cpp`` that uses flashlight:

::

   #include <iostream>

   #include <arrayfire.h>
   #include <flashlight/flashlight.h>

   /**
    * ###### #         ##    ####  #    # #      #  ####  #    # #####
    * #      #        #  #  #      #    # #      # #    # #    #   #
    * #####  #       #    #  ####  ###### #      # #      ######   #
    * #      #       ######      # #    # #      # #  ### #    #   #
    * #      #       #    # #    # #    # #      # #    # #    #   #
    * #      ####### #    #  ####  #    # ###### #  ####  #    #   #
    */
   int main() {
     fl::Variable v(af::array(1), true);
     auto result = v + 10;
     std::cout << "Hello World!" << std::endl;
     af::print("Array value is ", result.array());
     return 0;
   }

We can link flashlight with the following CMake configuration:

.. code-block:: shell
   
  # CMake 3.5.1+ is recommended
  cmake_minimum_required(VERSION 3.5.1)
  # C++ 11 is required
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  add_executable(myProject project.cpp)

  find_package(ArrayFire REQUIRED)
  # ...

  find_package(flashlight REQUIRED)
  # ...
  
  target_link_libraries(
    myProject
    PRIVATE
    ArrayFire::afcuda
    flashlight::flashlight # assumes flashlight was built with the CUDA backend
  )

The above will automatically link all flashlight backend-specific dependencies and will add the correct directories to the target's (``myProject``'s) include directories.
