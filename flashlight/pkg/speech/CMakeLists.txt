cmake_minimum_required(VERSION 3.10)

cmake_dependent_option(
  FL_BUILD_APP_ASR_SFX_SOX
  "Build libsox-based sound effects augmentation" OFF
  "FL_BUILD_APP_ASR" OFF
  )
cmake_dependent_option(
  FL_USE_SOX_STATIC
  "Statically link libsox with sound effects" OFF
  "NOT BUILD_SHARED_LIBS" OFF
  )

add_library(
  fl_pkg_speech
  ""
  )

target_link_libraries(
  fl_pkg_speech
  PUBLIC
  flashlight
  )

# ------------------------ Components ------------------------

# Common
include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)

# Criterion
include(${CMAKE_CURRENT_LIST_DIR}/criterion/CMakeLists.txt)

# Data
include(${CMAKE_CURRENT_LIST_DIR}/data/CMakeLists.txt)

# Decoder
include(${CMAKE_CURRENT_LIST_DIR}/decoder/CMakeLists.txt)

# Runtime
include(${CMAKE_CURRENT_LIST_DIR}/runtime/CMakeLists.txt)

# Augmentation
include(${CMAKE_CURRENT_LIST_DIR}/augmentation/CMakeLists.txt)

## --------------------------- Tests ---------------------------

## Build tests
if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif()

if (FL_CODE_COVERAGE)
  add_coverage_to_target(TARGET fl_pkg_speech)
endif()
