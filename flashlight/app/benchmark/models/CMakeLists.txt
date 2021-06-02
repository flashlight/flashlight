cmake_minimum_required(VERSION 3.10)

target_sources(
  flashlight-app-benchmark
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/AsrTransformer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/LmTransformer.cpp
  )
