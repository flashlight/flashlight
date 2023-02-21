cmake_minimum_required(VERSION 3.16)

include(FetchContent)

FetchContent_Declare(
  libsndfile
  GIT_REPOSITORY https://github.com/libsndfile/libsndfile.git
  GIT_TAG        1.1.0
  )

set(_BUILD_TESTING ${BUILD_TESTING}) # save

set(ENABLE_EXTERNAL_LIBS ON CACHE INTERNAL "Build libsndfile support for FLAC, Vorbis, and Opus")
set(ENABLE_MPEG OFF CACHE INTERNAL "Disable building libsndfile support for MPEG")
set(BUILD_TESTING OFF CACHE INTERNAL "Disable building libsndfile tests")

FetchContent_MakeAvailable(libsndfile)

set(BUILD_TESTING ${_BUILD_TESTING}) # restore
