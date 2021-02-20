cmake_minimum_required(VERSION 3.10)

# ----------------------------- Common -----------------------------
set(
  COMMON_SRCS
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DevicePtr.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Defines.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DynamicBenchmark.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Init.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Logging.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Histogram.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Plugin.cpp
)

if(FL_USE_CUDA)
  set(COMMON_SRCS ${COMMON_SRCS} ${CMAKE_CURRENT_LIST_DIR}/CudaUtils.cpp)
endif()
if(FL_USE_OPENCL)
  set(COMMON_SRCS ${COMMON_SRCS} ${CMAKE_CURRENT_LIST_DIR}/OpenClUtils.cpp)
endif()

target_sources(
  flashlight
  PRIVATE
  ${COMMON_SRCS}
)

# A native threading library is needed for ThreadPool
# (see flashlight/common/threadpool/ThreadPool.h)
find_package(Threads REQUIRED)

target_link_libraries(
  flashlight
  PUBLIC
  ${CMAKE_THREAD_LIBS_INIT} # threading library
  ${CMAKE_DL_LIBS} # for plugin
)

if (FL_USE_CUDA)
  target_link_libraries(flashlight PUBLIC ${CUDA_LIBRARIES})
  target_include_directories(flashlight PUBLIC ${CUDA_INCLUDE_DIRS})
endif()
if (FL_USE_OPENCL)
  target_link_libraries(flashlight PUBLIC OpenCL::OpenCL)
endif()
