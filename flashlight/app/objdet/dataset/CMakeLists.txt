cmake_minimum_required(VERSION 3.5.1)

target_sources(
  flashlight-app-obj-det
  PRIVATE
  #${CMAKE_CURRENT_LIST_DIR}/ImageDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Coco.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BoxUtils.cpp
)
