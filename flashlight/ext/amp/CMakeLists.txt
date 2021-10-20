cmake_minimum_required(VERSION 3.10)

if (FL_USE_CUDA)
target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/DynamicScaler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/Utils.cu
  )
else()
target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/DynamicScaler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/Utils.cpp
  )
endif()
