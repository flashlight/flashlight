cmake_minimum_required(VERSION 3.5.1)

target_sources(
  flashlight-app-asr
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/DecodeFlags.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DistributedFlags.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SharedFlags.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TrainFlags.cpp
  )
