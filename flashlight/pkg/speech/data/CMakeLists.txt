cmake_minimum_required(VERSION 3.16)

# ----------------------------- Dependencies -----------------------------
find_package(SndFile MODULE)
if (TARGET SndFile::sndfile)
  message(STATUS "libsndfile found.")
else ()
  if (FL_BUILD_STANDALONE)
    message(
      STATUS
      "libsndfile not found - will download and build from source.")
    include(${PROJECT_SOURCE_DIR}/cmake/BuildSndFile.cmake)
  else()
    message(FATAL_ERROR "Required dependency libsndfile not found")
  endif()
endif ()
if (FL_BUILD_STANDALONE)
  setup_install_find_module(${PROJECT_SOURCE_DIR}/cmake/FindSndFile.cmake)
  setup_install_find_module(${PROJECT_SOURCE_DIR}/cmake/FindOgg.cmake)
  setup_install_find_module(${PROJECT_SOURCE_DIR}/cmake/FindVorbis.cmake)
  setup_install_find_module(${PROJECT_SOURCE_DIR}/cmake/FindFLAC.cmake)
endif()

# ----------------------------- Lib -----------------------------
target_sources(
  fl_pkg_speech
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/FeatureTransforms.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ListFileDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Sound.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  )

target_link_libraries(
  fl_pkg_speech
  PRIVATE
  SndFile::sndfile
  )
