cmake_minimum_required(VERSION 3.16)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/AverageValueMeter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CountMeter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/EditDistanceMeter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/FrameErrorMeter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/MSEMeter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TimeMeter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TopKMeter.cpp
  )
