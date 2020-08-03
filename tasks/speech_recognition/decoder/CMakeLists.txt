cmake_minimum_required(VERSION 3.5.1)

target_sources(
  flashlight-task-speech-recognition
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/ConvLmModule.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  )
