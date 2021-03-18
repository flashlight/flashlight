# Find the rccl libraries
#
# The following variables are optionally searched for defaults
#  RCCL_ROOT_DIR: Base directory where all RCCL components are found
#  RCCL_INCLUDE_DIR: Directory where RCCL header is found
#  RCCL_LIB_DIR: Directory where RCCL library is found
#
# The following are set after configuration is done:
#  RCCL_FOUND
#  RCCL_INCLUDE_DIRS
#  RCCL_LIBRARIES
#
# The path hints include ROCM_TOOLKIT_ROOT_DIR seeing as some folks
# install RCCL in the same location as the ROCM toolkit.
# See https://github.com/caffe2/caffe2/issues/1601

set(RCCL_ROOT_DIR $ENV{RCCL_ROOT_DIR} CACHE PATH "Folder contains NVIDIA RCCL")

find_path(RCCL_INCLUDE_DIRS
  NAMES rccl.h
  HINTS
  ${RCCL_INCLUDE_DIR}
  ${RCCL_ROOT_DIR}
  ${RCCL_ROOT_DIR}/include
  ${ROCM_TOOLKIT_ROOT_DIR}/include)

IF ($ENV{USE_STATIC_RCCL})
  MESSAGE(STATUS "USE_STATIC_RCCL detected. Linking against static RCCL library")
  SET(RCCL_LIBNAME "librccl_static.a")
ELSE()
  SET(RCCL_LIBNAME "rccl")
ENDIF()

find_library(RCCL_LIBRARIES
  NAMES ${RCCL_LIBNAME}
  HINTS
  ${RCCL_LIB_DIR}
  ${RCCL_ROOT_DIR}
  ${RCCL_ROOT_DIR}/lib
  ${RCCL_ROOT_DIR}/lib/x86_64-linux-gnu
  ${RCCL_ROOT_DIR}/lib64
  ${ROCM_TOOLKIT_ROOT_DIR}/lib
  ${ROCM_TOOLKIT_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RCCL DEFAULT_MSG RCCL_INCLUDE_DIRS RCCL_LIBRARIES)

if(RCCL_FOUND)
  set (RCCL_HEADER_FILE "${RCCL_INCLUDE_DIRS}/rccl.h")
  message (STATUS "Determining RCCL version from the header file: ${RCCL_HEADER_FILE}")
  file (STRINGS ${RCCL_HEADER_FILE} RCCL_MAJOR_VERSION_DEFINED
    REGEX "^[ \t]*#define[ \t]+RCCL_MAJOR[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
  if (RCCL_MAJOR_VERSION_DEFINED)
    string (REGEX REPLACE "^[ \t]*#define[ \t]+RCCL_MAJOR[ \t]+" ""
      RCCL_MAJOR_VERSION ${RCCL_MAJOR_VERSION_DEFINED})
    message (STATUS "RCCL_MAJOR_VERSION: ${RCCL_MAJOR_VERSION}")
  endif ()
  message(STATUS "Found RCCL (include: ${RCCL_INCLUDE_DIRS}, library: ${RCCL_LIBRARIES})")
  mark_as_advanced(RCCL_ROOT_DIR RCCL_INCLUDE_DIRS RCCL_LIBRARIES)
  endif()
