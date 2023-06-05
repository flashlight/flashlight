cmake_minimum_required(VERSION 3.16)

add_library(fl_pkg_vision)

# Install lib and headers
list(APPEND INSTALLABLE_TARGETS fl_pkg_vision)
setup_install_headers(
  ${CMAKE_CURRENT_LIST_DIR}
  ${FL_INSTALL_INC_DIR_HEADER_LOC}/pkg)

target_link_libraries(
  fl_pkg_vision
  PUBLIC
  flashlight
  )

include(${CMAKE_CURRENT_LIST_DIR}/tensor/CMakeLists.txt)

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
