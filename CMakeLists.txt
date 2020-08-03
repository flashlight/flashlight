cmake_minimum_required(VERSION 3.5.1)

project(flashlight)

include(CTest)

# ----------------------------- Setup -----------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

include(InternalUtils)
set(INSTALLABLE_TARGETS "")

# ----------------------------- Configuration -----------------------------

option(FL_BUILD_TESTS "Build tests" ON)
option(FL_BUILD_EXAMPLES "Build examples" ON)
option(FL_BUILD_EXPERIMENTAL "Build internal experimental components" OFF)
option(FL_BUILD_SCRIPTS "Build internal scripts for wav2letter++" OFF)
option(FL_BUILD_RECIPES "Build recipes" ON)
option(FL_BUILD_CORE_ONLY "Build flashlight core only" OFF)
option(FL_BUILD_LIBRARIES_ONLY "Build flashlight libraries only" OFF)

# Flashlight backend
set(FL_BACKEND "CUDA" CACHE STRING "Backend with which to build flashlight")
# Select from exactly one backend
set_property(CACHE FL_BACKEND PROPERTY STRINGS CPU CUDA OPENCL)
# Map to flags
set(FLASHLIGHT_USE_CPU OFF)
set(FLASHLIGHT_USE_CUDA OFF)
set(FLASHLIGHT_USE_OPENCL OFF)
if (FL_BACKEND STREQUAL "CPU")
  set(FLASHLIGHT_USE_CPU ON)
elseif (FL_BACKEND STREQUAL "CUDA")
  set(FLASHLIGHT_USE_CUDA ON)
elseif (FL_BACKEND STREQUAL "OPENCL")
  set(FLASHLIGHT_USE_OPENCL ON)
else ()
  message(FATAL_ERROR "Invalid FLASHLIGHT backend specified")
endif ()

# ------------------------ Tests ------------------------

if (FL_BUILD_TESTS)
  enable_testing()
  include(TestUtils)
endif ()

# ------------------------ flashlight Libraries ------------------------
option(FL_LIBRARIES_USE_CUDA "Use CUDA in flashlight libraries build" ON)
option(FL_LIBRARIES_USE_KENLM "Use KenLM in flashlight libraries build" ON)
option(FL_LIBRARIES_USE_MKL "Use MKL in flashlight libraries build" ON)
option(FL_LIBRARIES_BUILD_FOR_PYTHON "Build Python bindings" OFF)

set(FL_COMPILE_DEFINITIONS
  $<$<BOOL:${FL_LIBRARIES_USE_CUDA}>:FL_LIBRARIES_USE_CUDA>
  $<$<BOOL:${FL_LIBRARIES_USE_MKL}>:FL_LIBRARIES_USE_MKL>
  $<$<BOOL:${FL_LIBRARIES_USE_KENLM}>:FL_LIBRARIES_USE_KENLM>
  )

set(FLASHLIGHT_LIB_DIR "${PROJECT_SOURCE_DIR}/libraries")
include(${FLASHLIGHT_LIB_DIR}/CMakeLists.txt)

setup_install_headers(${FLASHLIGHT_LIB_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC})

if (FL_BUILD_LIBRARIES_ONLY)
  message(STATUS "Libraries-only setup complete")
  setup_install_targets(INSTALL_TARGETS ${INSTALLABLE_TARGETS})
  return()
endif()

# --------------------------- flashlight ---------------------------
# Primary lib target. All non-library code is linked here, including tasks
add_library(flashlight "")

set_target_properties(
  flashlight
  PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 11
  )

# Link libraries to flashlight core
target_link_libraries(flashlight PUBLIC fl-libraries)

# ------------------------ Global External Dependencies ------------------------
# ArrayFire
find_package(ArrayFire REQUIRED)
if (ArrayFire_FOUND AND ArrayFire_VERSION VERSION_LESS 3.7.1)
  message(FATAL_ERROR "ArrayFire versions < 3.7.1 are no longer supported "
    "with flashlight. To build flashlight with a version of ArrayFire "
    "< 3.7.1, use commit <= 5518d91b7f4fd5b400cbc802cfbecc0df57836bd.")
endif()

if (ArrayFire_FOUND)
  message(STATUS "ArrayFire found (include: ${ArrayFire_INCLUDE_DIRS}, library: ${ArrayFire_LIBRARIES})")
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
endif()

# Find/download/build Cereal
if (TARGET CONAN_PKG::cereal)
  target_link_libraries(
    flashlight
    PUBLIC
    CONAN_PKG::cereal
    )
else()
   set(CEREAL_INSTALL_PATH ${FL_INSTALL_INC_DIR}/cereal)
   # If cereal is found in a user-defined location, use it rather than
   # downloading from source
   find_package(cereal)
   if (NOT cereal_FOUND)
     message(STATUS "cereal NOT found. Will download from source")
     include(${CMAKE_MODULE_PATH}/BuildCereal.cmake)
     add_dependencies(flashlight cereal)
     # Move cereal headers at install time
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

   target_include_directories(flashlight PUBLIC ${cereal_INCLUDE_DIRS})
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
endif()

if (FLASHLIGHT_USE_CPU)
  # MKL is not required to build the CPU backend (but is highly recommended
  # for improved performag ance) since MKL-DNN can be built with either mklml or
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

# Set up variables to link the right ArrayFire backend
if (FLASHLIGHT_USE_CPU)
  set(FL_AF_BACKEND "afcpu")
elseif (FLASHLIGHT_USE_CUDA)
  set(FL_AF_BACKEND "afcuda")
elseif (FLASHLIGHT_USE_OPENCL)
  set(FL_AF_BACKEND "afopencl")
else()
  message(FATAL_ERROR "flashlight backend ill-specified")
endif()

# Link ArrayFire to flashlight
target_link_libraries(flashlight INTERFACE ArrayFire::${FL_AF_BACKEND})

# ------------------------ flashlight Core ------------------------

set(FLASHLIGHT_CORE_DIR "${PROJECT_SOURCE_DIR}/flashlight")
include(${FLASHLIGHT_CORE_DIR}/CMakeLists.txt)
set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} flashlight)
# To remove the flashlight prefix path (so includes aren't
# flashlight/flashlight/flashlight.h, etc), glob all files below
# [root]/flashlight since the instal base path is flashlight
setup_install_headers(${FLASHLIGHT_CORE_DIR} ${FL_INSTALL_INC_DIR})

if (FL_BUILD_CORE_ONLY)
  message(STATUS "Core-only setup complete")
  setup_install_targets(INSTALL_TARGETS ${INSTALLABLE_TARGETS})
  return()
endif()

# ------------------------ Extensions ------------------------
set(FLASHLIGHT_EXT_DIR "${PROJECT_SOURCE_DIR}/extensions")
include(${FLASHLIGHT_EXT_DIR}/CMakeLists.txt)

# --------------------------- Tasks ---------------------------
set(FLASHLIGHT_TASKS_DIR "${PROJECT_SOURCE_DIR}/tasks")
include(${FLASHLIGHT_TASKS_DIR}/CMakeLists.txt)

# Fallthrough default install
setup_install_targets(INSTALL_TARGETS ${INSTALLABLE_TARGETS})
