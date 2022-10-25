cmake_minimum_required(VERSION 3.16)

include(FetchContent)

set(flashlight_sequence_URL https://github.com/flashlight/sequence.git)
set(flashlight_sequence_TAG dd9365ffa611f6fafa5273e154a64ae998172cd0)

FetchContent_Declare(
  flashlight-sequence
  GIT_REPOSITORY ${flashlight_sequence_URL}
  GIT_TAG        ${flashlight_sequence_TAG}
)

set(FL_SEQUENCE_USE_CUDA ${FL_USE_CUDA})
set(FL_SEQUENCE_BUILD_TESTS OFF)
FetchContent_MakeAvailable(flashlight-sequence)
add_library(flashlight::flashlight-sequence ALIAS flashlight-sequence)
