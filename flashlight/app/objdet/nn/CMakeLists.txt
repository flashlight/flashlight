cmake_minimum_required(VERSION 3.5.1)

target_sources(
  flashlight-app-obj-det
  PRIVATE
  # TODO (padentomasello) seperate src / header
  ${CMAKE_CURRENT_LIST_DIR}/Transformer.h
  ${CMAKE_CURRENT_LIST_DIR}/PositionalEmbeddingSine.cpp
)

