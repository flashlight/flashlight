cmake_minimum_required(VERSION 3.13)

add_library(fl_pkg_vision "")

target_link_libraries(fl_pkg_vision PUBLIC flashlight)

target_include_directories(fl_pkg_vision PUBLIC)

include(${CMAKE_CURRENT_LIST_DIR}/criterion/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/dataset/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/models/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/nn/CMakeLists.txt)

if (FL_CODE_COVERAGE)
  add_coverage_to_target(TARGET fl_pkg_vision)
endif()