# Try to find the KenLM library
#
# The following variables are optionally searched for defaults
#  KENLM_ROOT_DIR: Base directory where all KENLM components are found
#
# The following are set after configuration is done:
#  KENLM_FOUND
#  KENLM_REQUIRED_HEADERS - only the needed headers
#  KENLM_LIBRARIES
#

message(STATUS "Looking for KenLM")

find_library(
  KENLM_LIB
  kenlm
  HINTS
    ${KENLM_ROOT_DIR}/lib
    ${KENLM_ROOT_DIR}/build/lib
  PATHS
    $ENV{KENLM_ROOT_DIR}/lib
    $ENV{KENLM_ROOT_DIR}/build/lib
  )

find_library(
  KENLM_UTIL_LIB
  kenlm_util
  HINTS
    ${KENLM_ROOT_DIR}/lib
    ${KENLM_ROOT_DIR}/build/lib
  PATHS
    $ENV{KENLM_ROOT_DIR}/lib
    $ENV{KENLM_ROOT_DIR}/build/lib
  )

if(KENLM_LIB)
  message(STATUS "Using kenlm library found in ${KENLM_LIB}")
else()
  message(FATAL_ERROR "kenlm library not found; please set CMAKE_LIBRARY_PATH, KENLM_LIB or KENLM_ROOT_DIR environment variable")
endif()

if(KENLM_UTIL_LIB)
  message(STATUS "Using kenlm utils library found in ${KENLM_UTIL_LIB}")
else()
  message(FATAL_ERROR "kenlm utils library not found; please set CMAKE_LIBRARY_PATH, KENLM_UTIL_LIB or KENLM_ROOT_DIR environment variable")
endif()

# find a model header, then get the entire include directory. We need to do this because
# cmake consistently confuses other things along this path
find_file(KENLM_MODEL_HEADER
  model.hh
  HINTS
    ${KENLM_ROOT_DIR}/lm
    ${KENLM_ROOT_DIR}/include/kenlm/lm
  PATHS
    $ENV{KENLM_ROOT_DIR}/lm
    $ENV{KENLM_ROOT_DIR}/include/kenlm/lm
  )

if(KENLM_MODEL_HEADER)
  message(STATUS "kenlm model.hh found in ${KENLM_MODEL_HEADER}")
else()
  message(FATAL_ERROR "kenlm model.hh not found; please set CMAKE_INCLUDE_PATH, KENLM_MODEL_HEADER or KENLM_ROOT_DIR environment variable")
endif()
get_filename_component(KENLM_INCLUDE_LM ${KENLM_MODEL_HEADER} DIRECTORY)
get_filename_component(KENLM_INCLUDE_DIR ${KENLM_INCLUDE_LM} DIRECTORY)

set(KENLM_LIBRARIES ${KENLM_LIB} ${KENLM_UTIL_LIB})
set(KENLM_INCLUDE_DIRS ${KENLM_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(kenlm DEFAULT_MSG KENLM_INCLUDE_DIRS KENLM_LIBRARIES)

if (kenlm_FOUND)
  message(STATUS "Found kenlm (include: ${KENLM_INCLUDE_DIRS}, library: ${KENLM_LIBRARIES})")
  mark_as_advanced(KENLM_ROOT_DIR KENLM_INCLUDE_DIRS KENLM_LIBRARIES)
endif()
