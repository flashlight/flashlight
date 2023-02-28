cmake_minimum_required(VERSION 3.16)

include(FetchContent)

FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        8b5f1f37b5b75829fc72d38e7b5d4bcbf8a26d55
  )

FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
  FetchContent_Populate(stb)
endif()

set(stb_INCLUDE_DIRS ${stb_SOURCE_DIR})
