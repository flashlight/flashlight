cmake_minimum_required(VERSION 3.5.1)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Transforms.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Jpeg.cpp
)
