cmake_minimum_required(VERSION 3.16)

include(FetchContent)

FetchContent_Declare(
  gloo
  GIT_REPOSITORY https://github.com/facebookincubator/gloo.git
  GIT_TAG        4a5e339b764261d20fc409071dc7a8b8989aa195
  )

set(_USE_MPI ${USE_MPI}) # save

set(USE_MPI ON CACHE INTERNAL "Build Gloo with MPI support (required)")
FetchContent_MakeAvailable(gloo)

# gloo build doesn't add include directories as a target property...
target_include_directories(gloo PUBLIC
  $<BUILD_INTERFACE:${gloo_SOURCE_DIR}>
  $<BUILD_INTERFACE:${gloo_BINARY_DIR}> # config.h generated at cmake config time
  )

set(USE_MPI ${_USE_MPI}) # restore
