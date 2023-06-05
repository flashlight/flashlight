cmake_minimum_required(VERSION 3.16)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnAutogradExtension.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Conv2D.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Pool2D.cpp
  ${CMAKE_CURRENT_LIST_DIR}/RNN.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BatchNorm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DnnlUtils.cpp
)
