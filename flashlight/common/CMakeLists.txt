cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- Common -----------------------------
set(
  COMMON_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/common/Utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/common/DevicePtr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/common/Defines.cpp  
  )

if(FLASHLIGHT_USE_CUDA)
  set(COMMON_SRCS ${COMMON_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/common/CudaUtils.cpp)
endif()

add_library(
  Common
  INTERFACE
)

target_sources(
  Common
  INTERFACE
  $<BUILD_INTERFACE:${COMMON_SRCS}>
)

target_link_libraries(
  Common
  INTERFACE
  ${ArrayFire_LIBRARIES}
  ${CUDA_LIBRARIES}
)

target_include_directories(
  Common
  INTERFACE
  ${ArrayFire_INCLUDE_DIRS}
  ${CEREAL_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)
