cmake_minimum_required(VERSION 3.10)

target_sources(
  fl_pkg_speech
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/ConvLmModule.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DecodeMaster.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DecodeUtils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/PlGenerator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TranscriptionUtils.cpp
  )
