cmake_minimum_required(VERSION 3.10)

# ----------------------------- Sources -----------------------------
include(${CMAKE_CURRENT_LIST_DIR}/backends/CMakeLists.txt) # backends
include(${CMAKE_CURRENT_LIST_DIR}/passes/CMakeLists.txt) # passes

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Optimizer.cpp
)
