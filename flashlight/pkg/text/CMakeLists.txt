cmake_minimum_required(VERSION 3.10)

add_library(
  fl_pkg_text
  ""
  )

target_include_directories(
  fl_pkg_text
  PRIVATE
  ${PROJECT_SOURCE_DIR}
)

target_link_libraries(
  fl_pkg_text
  PUBLIC
  flashlight
  fl_lib_text
  )
# ------------------------ Components ------------------------

# Data
include(${CMAKE_CURRENT_LIST_DIR}/data/CMakeLists.txt)

## --------------------------- Tests ---------------------------

## Build tests
if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif()

if (FL_CODE_COVERAGE)
  add_coverage_to_target(TARGET fl_pkg_text)
endif()
