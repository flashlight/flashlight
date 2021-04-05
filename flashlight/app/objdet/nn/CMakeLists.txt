cmake_minimum_required(VERSION 3.10)

target_sources(
  flashlight-app-objdet
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Detr.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Transformer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/PositionalEmbeddingSine.cpp
)
