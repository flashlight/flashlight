# - Try to find MKLDNN
#
# The following variables are optionally searched for defaults
#  MKL_FOUND             : set to true if a library implementing the CBLAS interface is found
#  USE_MKLDNN
#
# The following are set after configuration is done:
#  MKLDNN_FOUND          : set to true if mkl-dnn is found.
#  MKLDNN_INCLUDE_DIR    : path to mkl-dnn include dir.
#  MKLDNN_LIBRARIES      : list of libraries for mkl-dnn

set(MKLDNN_LIBRARIES)
set(MKLDNN_INCLUDE_DIR)

# Check if there's any BLAS implementation available
find_package(BLAS)

if (NOT MKL_FOUND)
  message(FATAL_ERROR "Cannot build MKLDNN without MKL in current configuration.")
  # TODO(@jacobkahn) support building MKLDNN with standalone Intel libmklml
endif()

# Find headers
find_path(
  MKLDNN_INCLUDE_DIR
  mkldnn.hpp mkldnn.h
  PATHS
  ${MKLDNN_ROOT}
  PATH_SUFFIXES
  include
  HINTS
  ${MKLDNN_INC_DIR}
  )

if (MKLDNN_INCLUDE_DIR)
  message(STATUS "MKLDNN headers found in ${MKLDNN_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "MKLDNN headers not found; please set CMAKE_INCLUDE_PATH or
MKLDNN_ROOT")
endif()

# Find library
find_library(
  MKLDNN_LIBRARY
  mkldnn
  PATHS
  ${MKLDNN_ROOT}
  PATH_SUFFIXES
  lib
  HINTS
  ${MKLDNN_LIB_DIR}
)

if (MKLDNN_LIBRARY)
  message(STATUS "Using MKLDNN library found in ${MKLDNN_LIBRARY}")
else()
  message(FATAL_ERROR "MKLDNN library not found; please set CMAKE_LIBRARY_PATH o
r MKLDNN_LIBRARY")
endif()

set(MKLDNN_LIBRARIES ${MKLDNN_LIBRARY})

# TODO: link?
# Override OpenMP configuration for MKLDNN if MKL is found, so MKL OpenMP is used.
if (EXISTS "${MKL_LIBRARIES_gomp_LIBRARY}")
  set(MKLIOMP5LIB ${MKL_LIBRARIES_gomp_LIBRARY} CACHE STRING "Override MKL-DNN omp dependency" FORCE)
elseif(EXISTS "${MKL_LIBRARIES_iomp5_LIBRARY}")
  set(MKLIOMP5LIB ${MKL_LIBRARIES_iomp5_LIBRARY} CACHE STRING "Override MKL-DNN omp dependency" FORCE)
elseif(EXISTS "${MKL_LIBRARIES_libiomp5md_LIBRARY}")
  set(MKLIOMP5DLL ${MKL_LIBRARIES_libiomp5md_LIBRARY} CACHE STRING "Override MKL-DNN omp dependency" FORCE)
else(EXISTS "${MKL_LIBRARIES_gomp_LIBRARY}")
  set(MKLIOMP5LIB "" CACHE STRING "Override MKL-DNN omp dependency" FORCE)
  set(MKLIOMP5DLL "" CACHE STRING "Override MKL-DNN omp dependency" FORCE)
endif(EXISTS "${MKL_LIBRARIES_gomp_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKLDNN DEFAULT_MSG MKLDNN_LIBRARIES MKLDNN_INCLUDE_DIR)
