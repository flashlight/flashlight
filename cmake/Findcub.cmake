#
# Find cub headers
#
# Sets:
#  cub_INCLUDE_DIRS - location of headers
#  cub_FOUND        - truthy if cub was found.
#

find_path(cub_INCLUDE_DIRS cub.cuh PATH_SUFFIXES cub include PATHS ${cub_BASE_DIR})

mark_as_advanced(cub_INCLUDE_DIRS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cub DEFAULT_MSG cub_INCLUDE_DIRS)
