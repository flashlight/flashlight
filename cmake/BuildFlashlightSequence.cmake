cmake_minimum_required(VERSION 3.16)

include(FetchContent)

FetchContent_Declare(
  flashlight-sequence
  GIT_REPOSITORY https://github.com/flashlight/sequence.git
  GIT_TAG        5d288cd682542bd6b43f40af8019644af58bd1e6
)

set(FL_SEQUENCE_USE_CUDA ${FL_USE_CUDA} CACHE INTERNAL "Enable CUDA support in Flashlight Sequence")
set(FL_SEQUENCE_BUILD_TESTS OFF CACHE INTERNAL "Disable tests in Flashlight Sequence")
set(FL_SEQUENCE_BUILD_STANDALONE ${FL_BUILD_STANDALONE} CACHE INTERNAL "Set standalone build in Flashlight Sequence")
FetchContent_MakeAvailable(flashlight-sequence)
