cmake_minimum_required(VERSION 3.10)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Device.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DeviceManager.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DeviceType.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Stream.cpp
  )

if (FL_USE_CUDA)
  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/CUDADevice.cpp
    ${CMAKE_CURRENT_LIST_DIR}/CUDAStream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/CUDAUtils.cpp
    )
endif()
