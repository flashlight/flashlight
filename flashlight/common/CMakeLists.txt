cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- Common -----------------------------
set(
  COMMON_SRCS
  ${CMAKE_SOURCE_DIR}/flashlight/common/Utils.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/common/DevicePtr.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/common/Defines.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/common/Logging.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/common/Histogram.cpp
)

if(FLASHLIGHT_USE_CUDA)
  set(COMMON_SRCS ${COMMON_SRCS} ${CMAKE_SOURCE_DIR}/flashlight/common/CudaUtils.cpp)
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

# A native threading library is needed for ThreadPool
# (see flashlight/common/threadpool/ThreadPool.h)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(
  Common
  INTERFACE
  ${CMAKE_THREAD_LIBS_INIT} # threading library
)

target_include_directories(
  Common
  INTERFACE
  ${cereal_INCLUDE_DIRS} # pulled in - see cmake/BuildCereal.cmake
)

if (FLASHLIGHT_USE_CUDA)
  target_link_libraries(Common INTERFACE ${CUDA_LIBRARIES})
  target_include_directories(Common INTERFACE ${CUDA_INCLUDE_DIRS})
endif()
