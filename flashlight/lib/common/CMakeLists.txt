cmake_minimum_required(VERSION 3.10)

add_library(
  fl_lib_common
  ""
)

set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} fl_lib_common)

target_include_directories(
  fl_lib_common
  PRIVATE
  ${PROJECT_SOURCE_DIR}
  )

target_sources(
  fl_lib_common
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/String.cpp
  ${CMAKE_CURRENT_LIST_DIR}/System.cpp
  )

if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif ()
