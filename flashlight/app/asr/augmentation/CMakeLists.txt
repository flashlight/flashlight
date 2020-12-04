cmake_minimum_required(VERSION 3.10)

# ------------------------- Library -------------------------

add_library(
  flashlight-app-asr-sfx
  ${CMAKE_CURRENT_LIST_DIR}/SoundEffect.cpp
  )

target_link_libraries(
  flashlight-app-asr-sfx
  PUBLIC
  flashlight
  )
