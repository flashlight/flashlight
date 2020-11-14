cmake_minimum_required(VERSION 3.5.1)

target_sources(
  flashlight-app-asr
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/ConvLmModule.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DecodeMaster.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TranscriptionUtils.cpp
  )
