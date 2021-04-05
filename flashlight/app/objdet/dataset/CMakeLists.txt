cmake_minimum_required(VERSION 3.10)

target_sources(
  flashlight-app-objdet
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Coco.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Transforms.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BoxUtils.cpp
)
