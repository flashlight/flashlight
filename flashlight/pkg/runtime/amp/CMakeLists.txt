cmake_minimum_required(VERSION 3.10)

target_sources(
  fl_pkg_runtime
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/DynamicScaler.cpp
  )
