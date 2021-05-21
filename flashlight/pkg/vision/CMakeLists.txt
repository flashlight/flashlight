cmake_minimum_required(VERSION 3.10)

add_library(fl_pkg_vision "")

target_link_libraries(fl_pkg_vision PUBLIC flashlight)

include(${CMAKE_CURRENT_LIST_DIR}/criterion/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/dataset/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/models/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/nn/CMakeLists.txt)

if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif()

if (FL_CODE_COVERAGE)
  add_coverage_to_target(TARGET fl_pkg_vision)
endif()
