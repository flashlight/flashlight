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

# Don't build until cereal is properly included
add_dependencies(Common cereal)

target_sources(
  Common
  INTERFACE
  $<BUILD_INTERFACE:${COMMON_SRCS}>
)

target_link_libraries(
  Common
  INTERFACE
  ${ArrayFire_LIBRARIES}
)

target_include_directories(
  Common
  INTERFACE
  ${ArrayFire_INCLUDE_DIRS}
  ${CEREAL_INCLUDE_DIR} # pulled in - see cmake/BuildCereal.cmake
)

if (FLASHLIGHT_USE_CUDA)
  target_link_libraries(Common INTERFACE ${CUDA_LIBRARIES})
  target_include_directories(Common INTERFACE ${CUDA_INCLUDE_DIRS})
endif()
