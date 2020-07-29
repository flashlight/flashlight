# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.
#.rst:
# FindCUDNN
# -------
#
# Find CUDNN library
#
# Valiables that affect result:
# <VERSION>, <REQUIRED>, <QUIETLY>: as usual
#
# <EXACT> : as usual, plus we do find '5.1' version if you wanted '5'
#           (not if you wanted '5.0', as usual)
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``CUDNN_INCLUDE``
#   where to find cudnn.h.
# ``CUDNN_LIBRARY``
#   the libraries to link against to use CUDNN.
# ``CUDNN_FOUND``
#   If false, do not try to use CUDNN.
# ``CUDNN_VERSION``
#   Version of the CUDNN library we looked for

find_package(PkgConfig)
pkg_check_modules(PC_CUDNN QUIET CUDNN)

get_filename_component(__libpath_cudart "${CUDA_CUDART_LIBRARY}" PATH)

# We use major only in library search as major/minor is not entirely consistent among platforms.
# Also, looking for exact minor version of .so is in general not a good idea.
# More strict enforcement of minor/patch version is done if/when the header file is examined.
if(CUDNN_FIND_VERSION_EXACT)
  SET(__cudnn_ver_suffix ".${CUDNN_FIND_VERSION_MAJOR}")
  SET(__cudnn_lib_win_name cudnn64_${CUDNN_FIND_VERSION_MAJOR})
else()
  SET(__cudnn_lib_win_name cudnn64)
endif()

if(DEFINED ENV{CUDNN_ROOT_DIR})
  set(CUDNN_ROOT_DIR $ENV{CUDNN_ROOT_DIR} CACHE PATH "Folder contains NVIDIA cuDNN")
else()
  set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")
endif()

option(USE_STATIC_CUDNN "Use statically-linked cuDNN library" OFF)

if (USE_STATIC_CUDNN)
  # Windows not yet supported by FL and it is unknown if static cudnn works correctly on MacOS: supporting only Linux at the moment.
  # CMAKE_SYSTEM_NAME is supposed to resolve to "Linux", "Windows", or "Darwin"
  if (NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    MESSAGE(FATAL_ERROR "USE_STATIC_CUDNN only supported on Linux")
  endif()
  MESSAGE(STATUS "USE_STATIC_CUDNN detected. Linking against static CUDNN library")
  SET(CUDNN_LIBNAME "libcudnn_static.a" "cudnn_static")
  # culibos is needed to statically link with cudnn and usually installed in the regular cuda lib folders. On Linux:
  # /usr/local/cuda/lib64/libculibos.a
  # /usr/local/cuda/targets/x86_64-linux/lib/libculibos.a
  # https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html#culibos
  find_library(CULIBOS_LIBRARY
    NAMES culibos
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64
    DOC "culibos library")
  if ("${CULIBOS_LIBRARY}" STREQUAL "CULIBOS_LIBRARY-NOTFOUND")
    MESSAGE(FATAL_ERROR "CULIBOS not found")
  endif()
else()
  # shared lib:
  SET(CUDNN_LIBNAME libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name})
endif()
MESSAGE(STATUS "CUDNN libname: ${CUDNN_LIBNAME}")

find_library(CUDNN_LIBRARY
  NAMES ${CUDNN_LIBNAME}
  PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
  PATH_SUFFIXES lib lib64 bin
  DOC "CUDNN library." )

if(CUDNN_LIBRARY)
  SET(CUDNN_MAJOR_VERSION ${CUDNN_FIND_VERSION_MAJOR})
  set(CUDNN_VERSION ${CUDNN_MAJOR_VERSION})
  get_filename_component(__found_cudnn_root ${CUDNN_LIBRARY} PATH)
  find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    HINTS ${PC_CUDNN_INCLUDE_DIRS} ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_INCLUDE} ${__found_cudnn_root}
    PATH_SUFFIXES include
    DOC "Path to CUDNN include directory." )
endif()

if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
  file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)
  string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
    CUDNN_MAJOR_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
    CUDNN_MAJOR_VERSION "${CUDNN_MAJOR_VERSION}")
  string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
    CUDNN_MINOR_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
    CUDNN_MINOR_VERSION "${CUDNN_MINOR_VERSION}")
  string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
    CUDNN_PATCH_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
    CUDNN_PATCH_VERSION "${CUDNN_PATCH_VERSION}")
  set(CUDNN_VERSION ${CUDNN_MAJOR_VERSION}.${CUDNN_MINOR_VERSION})
endif()

if(CUDNN_MAJOR_VERSION)
  ## Fixing the case where 5.1 does not fit 'exact' 5.
  if(CUDNN_FIND_VERSION_EXACT AND NOT CUDNN_FIND_VERSION_MINOR)
    if("${CUDNN_MAJOR_VERSION}" STREQUAL "${CUDNN_FIND_VERSION_MAJOR}")
      set(CUDNN_VERSION ${CUDNN_FIND_VERSION})
    endif()
  endif()
else()
  # Try to set CUDNN version from config file
  set(CUDNN_VERSION ${PC_CUDNN_CFLAGS_OTHER})
endif()

find_package_handle_standard_args(
  CUDNN
  FOUND_VAR CUDNN_FOUND
  REQUIRED_VARS CUDNN_LIBRARY
  VERSION_VAR   CUDNN_VERSION
  )

if(CUDNN_FOUND)
  if(USE_STATIC_CUDNN)
    set(CUDNN_LIBRARIES ${CUDNN_LIBRARY} ${CULIBOS_LIBRARY})
  else()
    set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
  endif()
  set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
  set(CUDNN_DEFINITIONS ${PC_CUDNN_CFLAGS_OTHER})
  message(STATUS "Found CUDNN: (lib: ${CUDNN_LIBRARIES} include: ${CUDNN_INCLUDE_DIRS})")
endif()

