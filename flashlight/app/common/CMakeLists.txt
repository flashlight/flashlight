cmake_minimum_required(VERSION 3.10)

add_library(
  flashlight-app-common
  ""
)

target_sources(
  flashlight-app-common
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Runtime.cpp
)

target_link_libraries(
  flashlight-app-common
  PUBLIC
  ${GLOG_LIBRARIES}
  ${gflags_LIBRARIES}
  flashlight
)

target_include_directories(
  flashlight-app-common
  PUBLIC
  ${GLOG_INCLUDE_DIRS}
  ${gflags_INCLUDE_DIRS}
)

if (FL_CODE_COVERAGE)
  add_coverage_to_target(TARGET flashlight-app-common)
endif()
