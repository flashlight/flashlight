cmake_minimum_required(VERSION 3.5.1)

target_sources(
  flashlight-app-imclass
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Imagenet.cpp
)
