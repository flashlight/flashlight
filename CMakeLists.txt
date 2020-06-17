cmake_minimum_required(VERSION 3.5.1)

project(flashlight)

include(CTest)

# Conan support
set(CONAN_BUILD_INFO ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
if (EXISTS ${CONAN_BUILD_INFO})
  message(STATUS "Building with Conan.")
  include(${CONAN_BUILD_INFO})
  conan_basic_setup(TARGETS)
endif()

# ----------------------------- Setup -----------------------------
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR "${CMAKE_SOURCE_DIR}/flashlight") # module root

# Default directories for installation
set(FL_INSTALL_INC_DIR "include" CACHE PATH "Install path for headers")
set(FL_INSTALL_INC_DIR_HEADER_LOC ${FL_INSTALL_INC_DIR}/flashlight)
set(FL_INSTALL_LIB_DIR "lib" CACHE PATH "Install path for libraries")
set(FL_INSTALL_BIN_DIR "bin" CACHE PATH "Install path for binaries")
# Other assets
set(FL_INSTALL_ASSETS_BASE_DIR "share/flashlight")
set(FL_INSTALL_CMAKE_DIR "${FL_INSTALL_ASSETS_BASE_DIR}/cmake" CACHE PATH "Install path for CMake files")
set(FL_INSTALL_EXAMPLES_DIR "${FL_INSTALL_ASSETS_BASE_DIR}/examples" CACHE PATH "Install path for example files")
set(FL_INSTALL_DOC_DIR "${FL_INSTALL_ASSETS_BASE_DIR}/doc" CACHE PATH "Install path for documentation")

# ----------------------------- Configuration -----------------------------
# Flashlight backend
set(FLASHLIGHT_BACKEND "CUDA" CACHE STRING "Backend with which to build flashlight")
# Select from exactly one backend
set_property(CACHE FLASHLIGHT_BACKEND PROPERTY STRINGS UNIFIED CPU CUDA OPENCL)
# Map to flags
set(FLASHLIGHT_USE_UNIFIED OFF)
set(FLASHLIGHT_USE_CPU OFF)
set(FLASHLIGHT_USE_CUDA OFF)
set(FLASHLIGHT_USE_OPENCL OFF)
if (FLASHLIGHT_BACKEND STREQUAL "UNIFIED")
  # Currently unsupported
  message(FATAL_ERROR "Building FLASHLIGHT with the Unified backend is not currently supported")
  # set(FLASHLIGHT_USE_UNIFIED ON)
elseif (FLASHLIGHT_BACKEND STREQUAL "CPU")
  set(FLASHLIGHT_USE_CPU ON)
elseif (FLASHLIGHT_BACKEND STREQUAL "CUDA")
  set(FLASHLIGHT_USE_CUDA ON)
elseif (FLASHLIGHT_BACKEND STREQUAL "OPENCL")
  set(FLASHLIGHT_USE_OPENCL ON)
else ()
  message(FATAL_ERROR "Invalid FLASHLIGHT backend specified")
endif ()

# Distributed Training Backend
set(FL_BUILD_DISTRIBUTED ON CACHE BOOL "Whether to build and link the distributed backend with flashlight")
# If building with CUDA, use NCCL to on; if using CPU or OpenCL, use GLOO
set(USE_NCCL FALSE)
set(USE_GLOO FALSE)
if (FL_BUILD_DISTRIBUTED)
  if (FLASHLIGHT_USE_CUDA)
    set(USE_NCCL TRUE)
  elseif (FLASHLIGHT_USE_CPU OR FLASHLIGHT_USE_OPENCL OR FLASHLIGHT_USE_UNIFIED)
    set(USE_GLOO TRUE)
  endif ()
endif ()

# ------------------------ Global External Dependencies ------------------------
# ArrayFire
if (TARGET CONAN_PKG::arrayfire)
  message(STATUS "Found ArrayFire installed with Conan.")
else()
  find_package(ArrayFire REQUIRED)
  if (ArrayFire_FOUND AND ArrayFire_VERSION VERSION_LESS 3.7.1)
    message(FATAL_ERROR "ArrayFire versions < 3.7.1 are no longer supported "
      "with flashlight. To build flashlight with a version of ArrayFire "
      "< 3.7.1, use commit <= 5518d91b7f4fd5b400cbc802cfbecc0df57836bd.")
  endif()

  if (ArrayFire_FOUND)
    message(STATUS "ArrayFire found (include: ${ArrayFire_INCLUDE_DIRS}, library: ${ArrayFire_LIBRARIES})")
    if (FLASHLIGHT_USE_UNIFIED)
      # Set the AF_PATH environment variable to wherever the ArrayFire libs are
      # located so they can be loaded in the unified backend
      set(ENV{AF_PATH} ${ArrayFire_LIBRARIES})
    endif ()
  else()
    message(FATAL_ERROR "ArrayFire not found")
  endif()

  # Check the proper ArrayFire backend is present
  if (FLASHLIGHT_USE_CPU AND NOT ArrayFire_CPU_FOUND)
    message(FATAL_ERROR "ArrayFire CPU not found: cannot build CPU backend")
  elseif (FLASHLIGHT_USE_CUDA AND NOT ArrayFire_CUDA_FOUND)
    message(FATAL_ERROR "ArrayFire CUDA not found: cannot build CUDA backend")
  elseif (FLASHLIGHT_USE_OPENCL AND NOT ArrayFire_OpenCL_FOUND)
    message(FATAL_ERROR "ArrayFire OpenCL not found: cannot build OpenCL backend")
  elseif (FLASHLIGHT_USE_UNIFIED AND NOT ArrayFire_Unified_FOUND)
    message(FATAL_ERROR "ArrayFire Unified not found: cannot build unified backend")
  endif()
endif()

# Set up variables to link the right ArrayFire backend
if (FLASHLIGHT_USE_UNIFIED)
  set(FL_AF_BACKEND "af")
elseif (FLASHLIGHT_USE_CPU)
  set(FL_AF_BACKEND "afcpu")
elseif (FLASHLIGHT_USE_CUDA)
  set(FL_AF_BACKEND "afcuda")
elseif (FLASHLIGHT_USE_OPENCL)
  set(FL_AF_BACKEND "afopencl")
else()
  message(FATAL_ERROR "flashlight backend ill-specified")
endif()

# Link ArrayFire. Use a dummy interface library so the lib target can be
# exported and projects depending on projects can discern which AF
# (and FL) backend was built.
set(FL_BACKEND "BACKEND_${FLASHLIGHT_BACKEND}")
add_library(${FL_BACKEND} INTERFACE)
if (TARGET CONAN_PKG::arrayfire)
  target_link_libraries(${FL_BACKEND} INTERFACE CONAN_PKG::arrayfire)
else()
  target_link_libraries(${FL_BACKEND} INTERFACE ArrayFire::${FL_AF_BACKEND})
endif()


# Find/download/build Cereal
set(CEREAL_INSTALL_PATH ${FL_INSTALL_INC_DIR_HEADER_LOC}/cereal)
# If cereal is found in a user-defined location, use it rather than
if (TARGET CONAN_PKG::cereal)
  message(STATUS "Found cereal installed by Conan")
else ()
  # Try to find from source, else download
  find_package(cereal)
  if (NOT cereal_FOUND)
    message(STATUS "cereal NOT found. Will download from source")
    include(${CMAKE_MODULE_PATH}/BuildCereal.cmake)
    # Move cereal headers
    install(DIRECTORY ${CEREAL_SOURCE_DIR}/include/cereal
      DESTINATION ${CEREAL_INSTALL_PATH}
      COMPONENT cereal
      FILES_MATCHING
      PATTERN "*.hpp"
      PATTERN "*.h"
      PATTERN ".git" EXCLUDE
      )
    install(FILES ${CEREAL_SOURCE_DIR}/LICENSE ${CEREAL_SOURCE_DIR}/README.md
      DESTINATION ${CEREAL_INSTALL_PATH}
      )
  endif()
endif()

# -------------------- Locate Backend-specific Dependencies --------------------
# TODO: rather than conditionally searching for backend-specific dependencies,
# always search for all dependencies, and dynamically build all backends for
# which all required dependencies are found.

if (FLASHLIGHT_USE_CUDA)
   find_package(CUDA 9.2 QUIET) # CUDA 9.2 is required for >= ArrayFire 3.6.1
   if (CUDA_FOUND)
     message(STATUS "CUDA found (library: ${CUDA_LIBRARIES} include: ${CUDA_INCLUDE_DIRS})")
   else()
     message(STATUS "CUDA not found")
     message(FATAL_ERROR "CUDA required to build CUDA backend")
   endif()

   find_package(CUDNN 7.1 QUIET) # CUDNN 7.1 works with CUDA 9.2
   if (CUDNN_FOUND)
     message(STATUS "CUDNN found (library: ${CUDNN_LIBRARIES} include: ${CUDNN_INCLUDE_DIRS})")
   else()
     message(STATUS "CUDNN not found")
     message(FATAL_ERROR "CUDNN required to build CUDA backend")
   endif()
endif()

if (FLASHLIGHT_USE_CPU)
  # MKL is not required to build the CPU backend (but is highly recommended
  # for improved performance) since MKL-DNN can be built with either mklml or
  # MKL-DNN's internal GEMM implementation (see https://git.io/fhSMH). If MKL
  # isn't found, try to find mklml when finding MKL-DNN, and add it to
  # MKLDNN_LIBRARIES. If mklml can't be found, fall back to a system BLAS
  # library for GEMM. If no system lib is avaialble, use the internal MKL-DNN
  # GEMM implementation.
  find_package(MKL QUIET)
  if (MKL_FOUND)
    message(STATUS "MKL found")
  else()
    message(STATUS "MKL not found")
  endif()

  find_package(MKLDNN QUIET)
  if (MKLDNN_FOUND)
    message(STATUS "MKLDNN found")
  else()
    message(STATUS "MKLDNN not found")
    message(FATAL_ERROR "MKLDNN required to build CPU backend")
  endif()
endif()

if (FLASHLIGHT_USE_OPENCL)
  find_package(OpenCL)
  if (OpenCL_FOUND)
    message(STATUS "OpenCL found (library: ${OpenCL_LIBRARIES} include: ${OpenCL_INCLUDE_DIRS})")
  else()
    message(STATUS "OpenCL not found")
    if (FLASHLIGHT_USE_OPENCL)
      message(FATAL_ERROR "OpenCL required to build OpenCL backend")
    endif ()
  endif()
endif()


# -------------------------------- Main Library --------------------------------
add_library(flashlight "")

set_target_properties(
  flashlight
  PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 11
  )

set(
  FLASHLIGHT_MODULES
  ${FL_BACKEND}
  Autograd
  Common
  Dataset
  Distributed
  Memory
  Meter
  NN
  Optim
  )

# Build and link resources from `flashlight/contrib`.
set(FL_BUILD_CONTRIB ON CACHE BOOL
    "Build and link additional flashlight contrib resources.")
if (FL_BUILD_CONTRIB)
  message(STATUS "Will build flashlight contrib assets.")
  include(${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/contrib/CMakeLists.txt)
  set(FLASHLIGHT_MODULES ${FLASHLIGHT_MODULES} Contrib ${FLASHLIGHT_CONTRIB_LIBS})
endif()

# Build and link resources from `flashlight/experimental`.
set(FL_BUILD_EXPERIMENTAL OFF CACHE BOOL
    "Build and link additional internal flashlight experimental resources.")
if (FL_BUILD_EXPERIMENTAL)
  message(STATUS "Will build flashlight experimental assets.")
  include(${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/experimental/CMakeLists.txt)
  set(FLASHLIGHT_MODULES ${FLASHLIGHT_MODULES} FlExperimental
      ${FLASHLIGHT_EXPERIMENTAL_LIBS})
endif()

target_link_libraries(
  flashlight
  PUBLIC # export dependency library and include paths for each module
  ${FLASHLIGHT_MODULES}
  )

# Internal includes are impl defined as <flashlight...>
target_include_directories(
  flashlight
  PRIVATE
  ${CMAKE_SOURCE_DIR}
)

# -------------------------------- Components --------------------------------
# NOTE: each module is built as an interface library, but can't be built
# using add_subdirectory and installed with a common target (flashlight) -
# this is only recently supported in CMake:
# https://gitlab.kitware.com/cmake/cmake/merge_requests/2152. Because
# of this, each module must be included and added as a target to the
# main export.

# Autograd
include(${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/autograd/CMakeLists.txt)

# Common
include(${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/common/CMakeLists.txt)

# Dataset
include(${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/dataset/CMakeLists.txt)

# Dataset
include(${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/distributed/CMakeLists.txt)

# Memory
include(${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/memory/CMakeLists.txt)

# Meter
include(${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/meter/CMakeLists.txt)

# NN
include(${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/nn/CMakeLists.txt)

# Optim
include(${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/optim/CMakeLists.txt)

# ------------------------------- Install/Export -------------------------------

# Main target
install(
  TARGETS flashlight ${FLASHLIGHT_MODULES}
  EXPORT flashlightTargets
  # ARCHIVE DESTINATION ${FL_INSTALL_LIB_DIR}
  # INCLUDES DESTINATION ${FL_INSTALL_INC_DIR}
  COMPONENT flashlight
  PUBLIC_HEADER DESTINATION fl
  RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${FL_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${FL_INSTALL_LIB_DIR}
  FRAMEWORK DESTINATION framework
  INCLUDES DESTINATION ${FL_INSTALL_INC_DIR}
)

# Write and install targets file
install(
  EXPORT
  flashlightTargets
  NAMESPACE
  flashlight::
  DESTINATION
  ${FL_INSTALL_CMAKE_DIR}
  COMPONENT
  cmake)

# Move headers
install(
  DIRECTORY
  ${CMAKE_SOURCE_DIR}/flashlight/ # module headers in ./flashlight
  COMPONENT
  headers
  DESTINATION
  ${FL_INSTALL_INC_DIR_HEADER_LOC}
  FILES_MATCHING # preserve directory structure
  PATTERN  "*.h"
  )

# Move examples
# Don't build examples unless FL_BUILD_EXAMPLES is set, but always move them
install(
  DIRECTORY examples/
  DESTINATION ${FL_INSTALL_EXAMPLES_DIR}
  COMPONENT examples
  )

# Write config file (used by projects including fl, such as examples)
include(CMakePackageConfigHelpers)
set(INCLUDE_DIRS include)
set(CMAKE_DIR ${FL_INSTALL_CMAKE_DIR})
configure_package_config_file(
  ${CMAKE_MODULE_PATH}/flashlightConfig.cmake.in
  cmake/install/${FL_CONFIG_CMAKE_BUILD_DIR}/flashlightConfig.cmake
  INSTALL_DESTINATION
  ${FL_INSTALL_CMAKE_DIR}
  PATH_VARS INCLUDE_DIRS CMAKE_DIR
  )
install(FILES
  ${PROJECT_BINARY_DIR}/cmake/install/flashlightConfig.cmake
  DESTINATION ${FL_INSTALL_CMAKE_DIR}
  COMPONENT cmake
  )

# --------------------------- Configure Examples/Tests ---------------------------

# Build tests
option(FL_BUILD_TESTS "Build tests for flashlight" ON)
if (FL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/tests)
endif ()

# Build examples
option(FL_BUILD_EXAMPLES "Build examples for flashlight" ON)
if (FL_BUILD_EXAMPLES)
  add_subdirectory(${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/examples)
endif ()
