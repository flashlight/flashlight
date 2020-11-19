cmake_minimum_required(VERSION 3.10)

target_sources(
  flashlight-app-imgclass
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Imagenet.cpp
)
