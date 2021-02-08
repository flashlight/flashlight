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
  list(APPEND COMMON_SRCS ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/CudaUtils.cpp)
  if (FL_BUILD_PROFILING)
    list(APPEND COMMON_SRCS ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/Profile.cpp)
  endif()
endif()
if(FL_USE_OPENCL)
  set(COMMON_SRCS ${COMMON_SRCS} ${CMAKE_CURRENT_LIST_DIR}/OpenClUtils.cpp)
endif()
if(FL_USE_MIOPEN)
  set(
    COMMON_SRCS 
    ${COMMON_SRCS} 
    ${CMAKE_CURRENT_LIST_DIR}/MiOpenUtils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/OpenClUtils.cpp
    )
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
  # Try to find NVTX
  find_library(CUDA_NVTX_LIBRARIES
    NAMES nvToolsExt
    PATHS "${CUDA_TOOLKIT_ROOT_DIR}"
    ENV CUDA_PATH
    ENV CUDA_LIB_PATH
    ENV CUDA_HOME
    PATH_SUFFIXES lib64 lib
    NO_DEFAULT_PATH
    )

  if (NOT CUDA_NVTX_LIBRARIES)
    message(FATAL_ERROR "NVIDIA nvToolsExt (NVTX) lib not found; required.")
  else()
    message(STATUS "NVIDIA nvToolsExt (NVTX) found (lib: ${CUDA_NVTX_LIBRARIES})")
  endif()

  target_link_libraries(flashlight PUBLIC
    ${CUDA_LIBRARIES}
    ${CUDA_NVTX_LIBRARIES}
    )
  target_include_directories(flashlight PUBLIC ${CUDA_INCLUDE_DIRS})
endif()
if (FL_USE_OPENCL)
  target_link_libraries(flashlight PUBLIC OpenCL::OpenCL)

  add_executable(
    opencl_syntax_and_header
    ${PROJECT_SOURCE_DIR}/cmake/OpenCLSourceVerifierAndHeaderGen.cpp
  )

  target_include_directories(
    opencl_syntax_and_header
    PUBLIC
    ${OpenCL_INCLUDE_DIRS}
  )

  target_link_libraries(
    opencl_syntax_and_header
    PUBLIC
    ${OpenCL_LIBRARIES}
    ArrayFire::afopencl
  )
endif()
if (FL_USE_MIOPEN)
  target_link_libraries(flashlight PUBLIC ${MIOPENLIBRARIES})
  target_include_directories(flashlight PUBLIC ${MIOPEN_INCLUDE_DIRS})
endif()