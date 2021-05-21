cmake_minimum_required(VERSION 3.10)

target_sources(
  fl_pkg_speech
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Attention.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Logger.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SpeechStatMeter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Optimizer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Helpers.cpp
  )
