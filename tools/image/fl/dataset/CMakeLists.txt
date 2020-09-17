cmake_minimum_required(VERSION 3.5.1)

target_sources(
  flashlight-tools-image-fl
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Jpeg.cpp
)
