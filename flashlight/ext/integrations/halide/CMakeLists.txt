cmake_minimum_required(VERSION 3.10)

find_package(Halide CONFIG REQUIRED)

target_link_libraries(flashlight PUBLIC Halide::Runtime Halide::Tools)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/HalideInterface.cpp
  )

# if (FL_USE_CUDA)
#   target_sources(
#     flashlight
#     PRIVATE
#     ${CMAKE_CURRENT_LIST_DIR}/HalideCudaInterface.cpp
#     )
# endif()
