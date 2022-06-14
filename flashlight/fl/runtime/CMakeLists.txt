cmake_minimum_required(VERSION 3.10)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Device.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DeviceManager.cpp
  )

if (FL_USE_CUDA)
  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/CUDAUtils.cpp
    )
endif()
