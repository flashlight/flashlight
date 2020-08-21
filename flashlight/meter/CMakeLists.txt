cmake_minimum_required(VERSION 3.5.1)

set(
  METER_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/AverageValueMeter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CountMeter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/EditDistanceMeter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/FrameErrorMeter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/MSEMeter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TimeMeter.cpp
  )

target_sources(
  flashlight
  PUBLIC
  $<BUILD_INTERFACE:${METER_SOURCES}>
  )

