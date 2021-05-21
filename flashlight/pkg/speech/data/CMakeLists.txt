cmake_minimum_required(VERSION 3.10)

# ----------------------------- Dependencies -----------------------------
find_package(SndFile)
if (TARGET SndFile::sndfile)
  message(STATUS "libsndfile found.")
else ()
  if (FL_BUILD_STANDALONE)
    message(STATUS "libsndfile not found - will download and build from source.")
    include(${CMAKE_MODULE_PATH}/BuildSndFile.cmake)
    add_dependencies(fl_pkg_speech SndFile) # ExternalProject target
  else()
    message(FATAL_ERROR "Required dependency libsndfile not found")
  endif()
endif ()
if (FL_BUILD_STANDALONE)
  setup_install_find_module(${CMAKE_MODULE_PATH}/FindSndFile.cmake)
  setup_install_find_module(${CMAKE_MODULE_PATH}/FindOgg.cmake)
  setup_install_find_module(${CMAKE_MODULE_PATH}/FindVorbis.cmake)
  setup_install_find_module(${CMAKE_MODULE_PATH}/FindFLAC.cmake)
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
