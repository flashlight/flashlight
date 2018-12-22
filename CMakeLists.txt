cmake_minimum_required(VERSION 3.5.1)

project(flashlight)

include(CTest)

# ----------------------------- Setup -----------------------------
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR "${CMAKE_SOURCE_DIR}/flashlight") # module root

# ----------------------------- Configuration -----------------------------
# Arrayfire ML Backend
set(FLASHLIGHT_BACKEND "CUDA" CACHE STRING "Backend with which to build Arrayfire ML")
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
set(FL_BUILD_DISTRIBUTED "ON" CACHE STRING "Whether to build and link the distributed backend with flashlight")
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
find_package(ArrayFire 3.6.1 REQUIRED)
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

# Find cereal
find_package(cereal REQUIRED)
if (cereal_FOUND)
  message(STATUS "cereal found")
else()
  message(FATAL_ERROR "cereal not found")
endif()

# -------------------- Locate Backend-specific Dependencies --------------------
find_package(CUDA 9.2) # CUDA 9.2 is required for >= ArrayFire 3.6.1
if (CUDA_FOUND)
  message(STATUS "CUDA found (library: ${CUDA_LIBRARIES} include: ${CUDA_INCLUDE_DIRS})")
else()
  message(STATUS "CUDA not found")
  if (FLASHLIGHT_USE_CUDA)
    message(FATAL_ERROR "CUDA required to build CUDA backend")
  endif ()
endif()

find_package(CUDNN 7.2 QUIET) # CUDNN 7.2 works with CUDA 9.2
if (CUDNN_FOUND)
  message(STATUS "CUDNN found (library: ${CUDNN_LIBRARIES} include: ${CUDNN_INCLUDE_DIRS})")
else()
  message(STATUS "CUDNN not found")
  if (FLASHLIGHT_USE_CUDA)
    message(FATAL_ERROR "CUDNN required to build CUDA backend")
  endif ()
endif()

find_package(MKL)
if (MKL_FOUND)
  message(STATUS "MKL found")
else()
  message(STATUS "MKL not found")
  if (FLASHLIGHT_USE_CPU)
    message(FATAL_ERROR "MKL required to build CPU backend")
  endif()
endif()

find_package(MKLDNN)
if (MKLDNN_FOUND)
  message(STATUS "MKLDNN found")
else()
  message(STATUS "MKLDNN not found")
  if (FLASHLIGHT_USE_CPU)
    message(FATAL_ERROR "MKLDNN required to build CPU backend")
  endif()
endif()

find_package(OpenCL)
if (OpenCL_FOUND)
  message(STATUS "OpenCL found (library: ${OpenCL_LIBRARIES} include: ${OpenCL_INCLUDE_DIRS})")
else()
  message(STATUS "OpenCL not found")
  if (FLASHLIGHT_USE_OPENCL)
    message(FATAL_ERROR "OpenCL required to build OpenCL backend")
  endif ()
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
  Autograd
  Common
  Dataset
  Distributed
  Meter
  NN
  Optim
  )

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

# Link appropriate ArrayFire backend, set global compile time macros
if (FLASHLIGHT_USE_UNIFIED)
  # For flashlight, nn ops use CUDA in unified mode
  target_compile_definitions(
    flashlight
    PRIVATE
    BUILD_ARRAYFIRE_UNIFIED_BACKEND=1
    RUN_UNIFIED_BACKEND_TESTS=1)
  target_link_libraries(flashlight ArrayFire::af)
elseif (FLASHLIGHT_USE_CPU)
  target_compile_definitions(flashlight PRIVATE FLASHLIGHT_BUILD_CPU_BACKEND=1)
  target_link_libraries(flashlight PRIVATE ArrayFire::afcpu)
elseif (FLASHLIGHT_USE_CUDA)
  target_compile_definitions(flashlight PRIVATE FLASHLIGHT_BUILD_CUDA_BACKEND=1)
  target_link_libraries(flashlight PRIVATE ArrayFire::afcuda)
elseif (FLASHLIGHT_USE_OPENCL)
  target_compile_definitions(flashlight PRIVATE FLASHLIGHT_BUILD_OPENCL_BACKEND=1)
  target_link_libraries(flashlight PRIVATE ArrayFire::afopencl)
else()
  message(FATAL_ERROR "flashlight backend ill-specified")
endif()

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

# Meter
include(${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/meter/CMakeLists.txt)

# NN
include(${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/nn/CMakeLists.txt)

# Optim
include(${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/optim/CMakeLists.txt)

# ------------------------------- Install/Export -------------------------------

# Default directories for installation
set(FL_INSTALL_INC_DIR "include" CACHE PATH "Install path for headers")
set(FL_INSTALL_LIB_DIR "lib" CACHE PATH "Install path for libraries")
set(FL_INSTALL_BIN_DIR "bin" CACHE PATH "Install path for binaries")
# Other assets
set(FL_INSTALL_ASSETS_BASE_DIR "share/flashlight")
set(FL_INSTALL_CMAKE_DIR "${FL_INSTALL_ASSETS_BASE_DIR}/cmake" CACHE PATH "Install path for CMake files")
set(FL_INSTALL_EXAMPLES_DIR "${FL_INSTALL_ASSETS_BASE_DIR}/examples" CACHE PATH "Install path for example files")
set(FL_INSTALL_DOC_DIR "${FL_INSTALL_ASSETS_BASE_DIR}/doc" CACHE PATH "Install path for documentation")

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
  ${FL_INSTALL_INC_DIR}/flashlight
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
  add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
endif ()

# Build examples
option(FL_BUILD_EXAMPLES "Build examples for flashlight" ON)
if (FL_BUILD_EXAMPLES)
  add_subdirectory(${CMAKE_SOURCE_DIR}/examples)
endif ()
