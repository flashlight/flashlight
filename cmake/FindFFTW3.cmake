# - Find the FFTW library
#
# Usage:
#   find_package(FFTW3 [REQUIRED] [QUIET] )
#
# Provides the following config targets:
# - FFTW3::fftw3
#
# It sets the following variables:
#   FFTW3_FOUND               ... true if fftw is found on the system
#   FFTW3_LIBRARIES           ... full path to fftw library
#   FFTW3_INCLUDES            ... fftw include directory
#
# The following variables will be checked by the function
#   FFTW3_USE_STATIC_LIBS    ... if true, only static libraries are found
#   FFTW3_ROOT               ... if set, the libraries are exclusively searched
#                               under this path
#   FFTW3_LIBRARY            ... fftw library to use
#   FFTW3_INCLUDE_DIR        ... fftw include directory
#

find_package(FFTW3 CONFIG)

if (NOT FFTW3_FOUND)
  # If environment variable FFTWDIR is specified, it has same effect as FFTW3_ROOT
  if (NOT FFTW3_ROOT AND ENV{FFTWDIR} )
    set( FFTW3_ROOT $ENV{FFTWDIR} )
  endif()
  # Check if we can use PkgConfig
  find_package(PkgConfig)
  #Determine from PKG
  if (PKG_CONFIG_FOUND AND NOT FFTW3_ROOT )
    pkg_check_modules( PKG_FFTW QUIET "fftw3" )
    message(STATUS "FindFFTW using pkgconfig: FOUND=${PKG_FFTW_FOUND} LIBRARIES=${PKG_FFTW_LIBRARIES} LIBRARY_DIRS=${PKG_FFTW_LIBRARY_DIRS} LIBDIR=${PKG_FFTW_LIBDIR} LINK_LIBRARIES=${PKG_FFTW_LINK_LIBRARIES}")
    message(STATUS "FindFTTW using pkgconfig: INCLUDE_DIRS=${PKG_FFTW_INCLUDE_DIRS} INCLUDEDIR=${PKG_FFTW_INCLUDEDIR}")
    # Note: PKG_FFTW_LIBRARY_DIRS and PKG_FFTW_INCLUDE_DIRS are empty with pkg-config 0.29/ubuntu16/cmake3.13. Using PKG_FFTW_LIBDIR / PKG_FFTW_INCLUDEDIR instead:
    if (NOT PKG_FFTW_LIBRARY_DIRS)
      set(PKG_FFTW_LIBRARY_DIRS ${PKG_FFTW_LIBDIR})
    endif()
    if (NOT PKG_FFTW_INCLUDE_DIRS)
      set(PKG_FFTW_INCLUDE_DIRS ${PKG_FFTW_INCLUDEDIR})
    endif()
  endif()
  #Check whether to search static or dynamic libs
  set( CMAKE_FIND_LIBRARY_SUFFIXES_SAV ${CMAKE_FIND_LIBRARY_SUFFIXES} )
  if (${FFTW3_USE_STATIC_LIBS} )
    set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX} )
  else()
    set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX} )
  endif()
  if (FFTW3_ROOT )
    #find libs
    find_library(
      FFTW3_LIB
      NAMES "fftw3"
      PATHS ${FFTW3_ROOT}
      PATH_SUFFIXES "lib" "lib64"
      NO_DEFAULT_PATH
      )
    find_library(
      FFTWF_LIB
      NAMES "fftw3f"
      PATHS ${FFTW3_ROOT}
      PATH_SUFFIXES "lib" "lib64"
      NO_DEFAULT_PATH
      )
    find_library(
      FFTWL_LIB
      NAMES "fftw3l"
      PATHS ${FFTW3_ROOT}
      PATH_SUFFIXES "lib" "lib64"
      NO_DEFAULT_PATH
      )
    #find includes
    find_path(
      FFTW3_INCLUDES
      NAMES "fftw3.h"
      PATHS ${FFTW3_ROOT}
      PATH_SUFFIXES "include"
      NO_DEFAULT_PATH
      )
  else()
    find_library(
      FFTW3_LIB
      NAMES "fftw3"
      PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
      )
    find_library(
      FFTWF_LIB
      NAMES "fftw3f"
      PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
      )
    find_library(
      FFTWL_LIB
      NAMES "fftw3l"
      PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
      )
    find_path(
      FFTW3_INCLUDES
      NAMES "fftw3.h"
      PATHS ${PKG_FFTW3_INCLUDE_DIRS} ${INCLUDE_INSTALL_DIR}
      )
  endif (FFTW3_ROOT )
  set(FFTW3_LIBRARIES ${FFTW3_LIB} ${FFTWF_LIB})
  if(FFTWL_LIB)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTWL_LIB})
  endif()
  set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV} )
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(FFTW3 DEFAULT_MSG
    FFTW3_INCLUDES FFTW3_LIBRARIES)
  mark_as_advanced(FFTW3_INCLUDES FFTW3_LIBRARIES FFTW3_LIB FFTWF_LIB FFTWL_LIB)
  if (FFTW3_FOUND)
    add_library(FFTW3::fftw3 UNKNOWN IMPORTED)
    set_target_properties(FFTW3::fftw3 PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDES}"
      IMPORTED_LOCATION "${FFTW3_LIB}"
      )
    add_library(FFTW3::fftw3f UNKNOWN IMPORTED)
    set_target_properties(FFTW3::fftw3f PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDES}"
      IMPORTED_LOCATION "${FFTWF_LIB}"
      )
    add_library(FFTW3::fftw3l UNKNOWN IMPORTED)
    set_target_properties(FFTW3::fftw3l PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDES}"
      IMPORTED_LOCATION "${FFTWL_LIB}"
      )
  endif()
endif()
