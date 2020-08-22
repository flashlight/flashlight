cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- Common -----------------------------
set(
  COMMON_SRCS
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DevicePtr.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Defines.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Logging.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Histogram.cpp
)

if(FLASHLIGHT_USE_CUDA)
  set(COMMON_SRCS ${COMMON_SRCS} ${CMAKE_CURRENT_LIST_DIR}/CudaUtils.cpp)
endif()

target_sources(
  flashlight
  PUBLIC
  $<BUILD_INTERFACE:${COMMON_SRCS}>
)

# A native threading library is needed for ThreadPool
# (see flashlight/common/threadpool/ThreadPool.h)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(
  flashlight
  PUBLIC
  ${CMAKE_THREAD_LIBS_INIT} # threading library
)

if (FLASHLIGHT_USE_CUDA)
  target_link_libraries(flashlight PUBLIC ${CUDA_LIBRARIES})
  target_include_directories(flashlight PUBLIC ${CUDA_INCLUDE_DIRS})
endif()
