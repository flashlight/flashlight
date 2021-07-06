cmake_minimum_required(VERSION 3.10)


add_library(
  fl_lib_set
  ""
  )

set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} fl_lib_set)

target_include_directories(
  fl_lib_set
  PRIVATE
  ${PROJECT_SOURCE_DIR}
  )

target_sources(
  fl_lib_set
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Hungarian.cpp
  )

if (FL_CODE_COVERAGE)
  add_coverage_to_target(TARGET fl_lib_set)
endif()
