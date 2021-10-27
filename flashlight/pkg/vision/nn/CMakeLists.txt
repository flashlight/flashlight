cmake_minimum_required(VERSION 3.10)

target_sources(
  fl_pkg_vision
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/FrozenBatchNorm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/PositionalEmbeddingSine.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Transformer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/VisionTransformer.cpp
)
