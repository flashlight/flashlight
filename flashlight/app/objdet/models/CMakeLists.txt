cmake_minimum_required(VERSION 3.10)

target_sources(
  flashlight-app-objdet
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Resnet50Backbone.cpp
)
