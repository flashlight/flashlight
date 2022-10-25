cmake_minimum_required(VERSION 3.16)

include(FetchContent)

set(flashlight_sequence_URL https://github.com/flashlight/sequence.git)
set(flashlight_sequence_TAG 7ef5ab16cc609c9b7139f7961f2fdf37c637d2b6)

FetchContent_Declare(
  flashlight-sequence
  GIT_REPOSITORY ${flashlight_sequence_URL}
  GIT_TAG        ${flashlight_sequence_TAG}
)

set(FL_SEQUENCE_USE_CUDA ${FL_USE_CUDA})
set(FL_SEQUENCE_BUILD_TESTS OFF)
FetchContent_MakeAvailable(flashlight-sequence)
add_library(flashlight::flashlight-sequence ALIAS flashlight-sequence)
