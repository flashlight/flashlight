cmake_minimum_required(VERSION 3.16)

include(FetchContent)

FetchContent_Declare(
  flashlight-text
  GIT_REPOSITORY https://github.com/flashlight/text.git
  GIT_TAG        bb7809bfb6399a2e94a92cb25c83d15e45aa386f # > v0.0.2
)

# KenLM support is required for Flashlight's speech package
set(FL_TEXT_USE_KENLM ON CACHE INTERNAL "Enable KenLM support in Flashlight Text")
set(FL_TEXT_BUILD_TESTS OFF CACHE INTERNAL "Disable tests in Flashlight Text")
set(FL_TEXT_BUILD_STANDALONE ${FL_BUILD_STANDALONE} CACHE INTERNAL "Set standalone build in Flashlight Text")

FetchContent_MakeAvailable(flashlight-text)
