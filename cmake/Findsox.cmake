# Find the sox libraries
# Support shared libsox since the static libsox is dependent on a
# number of other libraries.
#
# The following variables affect the behavior of the macros in the script:
#  sox_ROOT_DIR: Base directory where all SOX components are found
#  sox_INCLUDE_DIR: Directory where SOX header is found
#  sox_LIB_DIR: Directory where SOX library is found
#
# The following are set after configuration is done:
#  sox_FOUND
#  sox_INCLUDE_DIRS
#  sox_LIBRARIES
#

if (NOT SOX_FOUND)

  # Search using package config
  find_package(PkgConfig)
  if (PKG_CONFIG_FOUND)
       pkg_check_modules(PKG_SOX QUIET libsox-dev)
       if (PKG_SOX_FOUND)
        message(STATUS "PkgConfig found libsox-dev (include: ${PKG_SOX_INCLUDE_DIRS}, library: ${PKG_SOX_LIBRARIES})")
        set(sox_INCLUDE_DIRS ${PKG_SOX_INCLUDE_DIRS})
        set(sox_LIBRARIES ${PKG_SOX_LIBRARIES})
        set(SOX_FOUND TRUE)
     endif()
  endif()

  if (NOT SOX_FOUND)
    set(sox_ROOT_DIR $ENV{sox_ROOT_DIR} CACHE PATH "Folder contains the sox library")

    find_path(sox_INCLUDE_DIRS
      NAMES sox.h
      PATHS ${sox_INCLUDE_DIR} ${sox_ROOT_DIR}
      PATH_SUFFIXES include
      )

    SET(sox_LIBNAME "libsox${CMAKE_SHARED_LIBRARY_SUFFIX}")

    find_library(sox_LIBRARIES
      NAMES ${sox_LIBNAME}
      PATHS ${sox_LIB_DIR} ${sox_ROOT_DIR}
      PATH_SUFFIXES "lib" "lib64"
      )

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(SOX DEFAULT_MSG sox_INCLUDE_DIRS sox_LIBRARIES)

    if(SOX_FOUND)
      set (sox_HEADER_FILE "${sox_INCLUDE_DIRS}/sox.h")
      message(STATUS "Found SOX (include: ${sox_INCLUDE_DIRS}, library: ${sox_LIBRARIES})")
      mark_as_advanced(sox_ROOT_DIR sox_INCLUDE_DIRS sox_LIBRARIES)
    endif()
  endif()
endif()
