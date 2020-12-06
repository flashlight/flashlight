cmake_minimum_required(VERSION 3.10)

# ----------------------------- Dependencies -----------------------------
find_package(SndFile REQUIRED)
if (TARGET SndFile::sndfile)
  message(STATUS "libsndfile found.")
  setup_install_find_module(${CMAKE_MODULE_PATH}/FindSndFile.cmake)
  setup_install_find_module(${CMAKE_MODULE_PATH}/FindOgg.cmake)
  setup_install_find_module(${CMAKE_MODULE_PATH}/FindVorbis.cmake)
  setup_install_find_module(${CMAKE_MODULE_PATH}/FindFLAC.cmake)
else ()
  message(FATAL_ERROR "Required dependency libsndfile not found.")
endif ()

# ----------------------------- Lib -----------------------------
target_sources(
  flashlight-app-asr
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Featurize.cpp
  ${CMAKE_CURRENT_LIST_DIR}/FeatureTransforms.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ListFileDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Sound.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SpeechSample.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Dataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BlobsDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ListFilesDataset.cpp
  )

target_link_libraries(
  flashlight-app-asr
  PUBLIC
  flashlight-app-asr-sfx
  PRIVATE
  SndFile::sndfile
  )
