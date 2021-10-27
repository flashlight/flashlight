cmake_minimum_required(VERSION 3.10)

target_sources(
  fl_pkg_vision
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Detr.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Resnet.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Resnet50Backbone.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ResnetFrozenBatchNorm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ViT.cpp
)
