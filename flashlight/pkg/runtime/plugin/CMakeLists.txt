cmake_minimum_required(VERSION 3.10)

target_sources(
  fl_pkg_runtime
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/ModulePlugin.cpp
  )

# Plugin Compiler - only run if a plugin path is passed
if (NOT ${FL_PLUGIN_MODULE_SRC_PATH} STREQUAL "")
  include(${CMAKE_CURRENT_LIST_DIR}/plugincompiler/CMakeLists.txt)
endif()
