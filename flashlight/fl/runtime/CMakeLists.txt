cmake_minimum_required(VERSION 3.16)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Device.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DeviceManager.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DeviceType.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Stream.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SynchronousStream.cpp
  )

if (FL_USE_CUDA)
  # Find cudart library
  if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    target_link_libraries(flashlight PUBLIC $<BUILD_INTERFACE:CUDA::cudart>)
  else()
    # TODO: switch to CMAKE_CUDA_RUNTIME_LIBRARY when requiring CMake >= 3.17
    find_library(CUDART_LIBRARY cudart PATHS ${CUDA_LIBRARIES} ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    if (NOT CUDART_LIBRARY)
      message(FATAL_ERROR "Could not find cudart library - required if using CUDA")
    endif()
    target_include_directories(
      flashlight
      PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
      )
    target_link_libraries(flashlight PUBLIC ${CUDART_LIBRARY})
  endif()

  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/CUDADevice.cpp
    ${CMAKE_CURRENT_LIST_DIR}/CUDAStream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/CUDAUtils.cpp
    )
endif()

target_compile_definitions(
  flashlight
  PUBLIC
  "-DNO_CUDA_STREAM_DESTROY_EVENT"
  )
