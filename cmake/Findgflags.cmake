# - Try to find gflags
#
# The following variables are optionally searched for defaults
#  gflags_ROOT_DIR:            Base directory where all gflags components are found
#
# The following are set after configuration is done:
#  gflags_FOUND
#  gflags_INCLUDE_DIRS
#  gflags_LIBRARIES

include(FindPackageHandleStandardArgs)

set(gflags_ROOT_DIR "" CACHE PATH "Folder contains Gflags")

find_path(gflags_INCLUDE_DIR gflags/gflags.h
    PATHS ${gflags_ROOT_DIR})

find_library(gflags_LIBRARY NAMES gflags gflags_debug)

find_package_handle_standard_args(gflags DEFAULT_MSG gflags_INCLUDE_DIR gflags_LIBRARY)

if(gflags_FOUND)
  set(gflags_INCLUDE_DIRS ${gflags_INCLUDE_DIR})
  set(gflags_LIBRARIES ${gflags_LIBRARY})
  message(STATUS "Found gflags  (include: ${gflags_INCLUDE_DIR}, library: ${gflags_LIBRARY})")
  mark_as_advanced(gflags_LIBRARY_DEBUG gflags_LIBRARY_RELEASE
    gflags_LIBRARY gflags_INCLUDE_DIR gflags_ROOT_DIR)
endif()
