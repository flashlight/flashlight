cmake_minimum_required(VERSION 3.16)

target_sources(
  benchmark
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/AsrTransformer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/LmTransformer.cpp
  )
