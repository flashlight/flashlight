cmake_minimum_required(VERSION 3.16)

include(FetchContent)

FetchContent_Declare(
  flashlight-text
  GIT_REPOSITORY https://github.com/flashlight/text.git
  GIT_TAG        v0.0.1
)

# KenLM support is required for Flashlight's speech package
set(FL_TEXT_USE_KENLM ON CACHE INTERNAL "Enable KenLM support in Flashlight Text")
set(FL_TEXT_BUILD_TESTS OFF CACHE INTERNAL "Disable tests in Flashlight Text")
set(FL_TEXT_BUILD_STANDALONE ${FL_BUILD_STANDALONE} CACHE INTERNAL "Set standalone build in Flashlight Text")

FetchContent_MakeAvailable(flashlight-text)
