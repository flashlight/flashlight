cmake_minimum_required(VERSION 3.5.1)

target_sources(
  flashlight-app-asr
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Logger.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SpeechStatMeter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Optimizer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Helpers.cpp
  )
