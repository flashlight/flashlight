cmake_minimum_required(VERSION 3.10)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Resnet.cpp
  ${CMAKE_CURRENT_LIST_DIR}/FrozenBatchNorm.cpp
)
