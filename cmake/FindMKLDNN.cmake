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
  message(STATUS "Attempting to build MKLDNN without MKL in current configuration.")
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
  message(STATUS "MKLDNN headers not found; please set CMAKE_INCLUDE_PATH or MKLDNN_ROOT, or MKLDNN_INC_DIR")
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
  message(STATUS "MKLDNN library not found; please set CMAKE_LIBRARY_PATH or MKLDNN_ROOT, or MKLDNN_LIB_DIR")
endif()

set(MKLDNN_LIBRARIES ${MKLDNN_LIBRARY})

# In order of preference, try to find and use MKL, mklml, then any system BLAS lib
if (MKL_FOUND)
  # Add MKL to MKLDNN deps if found
  message(STATUS "Using MKL with MKL-DNN")
  list(APPEND MKLDNN_LIBRARIES ${MKL_LIBRARIES})
  list(APPEND MKLDNN_INCLUDE_DIR ${MKL_INCLUDE_DIR})
else()
  message(STATUS "MKL not found; trying to fall back to mklml library")
  # MKL isn't found, so use the mini-MKL library (mklml) with MKL-DNN
  find_library(
    MKLML_LIBRARY
    NAMES
      mklml
      mklml_intel
    PATHS
    ${MKLDNN_ROOT}
    PATH_SUFFIXES
    lib
    HINTS
    ${MKLDNN_LIB_DIR}
    )
  # Find mklml headers. Perform this check anyways even though it's not clear
  # if they're being used, and they're not moved to the install dir on the
  # MKL-DNN install step.
  find_path(
    MKLML_INCLUDE_DIR
    mkl.h mkl_dnn_types.h
    PATHS
    ${MKLDNN_ROOT}
    PATH_SUFFIXES
    include
    external
    HINTS
    ${MKLDNN_INC_DIR}
    ${MKLML_INC_DIR}
    )
  if (MKLML_INCLUDE_DIR)
    message(STATUS "Found mklml headers: ${MKLML_INCLUDE_DIR}")
    list(APPEND MKLDNN_INCLUDE_DIR ${MKLML_INCLUDE_DIR})
  else()
    message(STATUS "Using MKL-DNN without mklml headers")
  endif()
  
  if (MKLML_LIBRARY)
    message(STATUS "Found libmklml: ${MKLML_LIBRARY}")
    message(STATUS "Using mklml with MKL-DNN")
    list(APPEND MKLDNN_LIBRARIES ${MKLML_LIBRARY})
  else()
    # If we still can't find mklml, look for any viable BLAS library as a last resort
    message(STATUS "mklml not found; trying to fall back to system BLAS library")
    if (BLAS_FOUND)
      message(STATUS "BLAS libraries found: ${BLAS_LIBRARIES}")
    else()
      # Build without a GEMM implementation
      message(STATUS "No GEMM implementation found - MKL-DNN will use internal GEMM implementation")
    endif()
  endif()
endif ()

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
