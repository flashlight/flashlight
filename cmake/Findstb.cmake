#
# Find stb headers
#
# Sets:
#  stb_INCLUDE_DIRS - location of headers
#  stb_FOUND        - truthy if stb was found.
#

find_path(stb_INCLUDE_DIRS stb_image.h PATH_SUFFIXES include PATHS ${stb_BASE_DIR})

mark_as_advanced(stb_INCLUDE_DIRS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(stb DEFAULT_MSG stb_INCLUDE_DIRS)
