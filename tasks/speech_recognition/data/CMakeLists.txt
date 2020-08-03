cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- Dependencies -----------------------------

# Sndfile
# The SndFile build/library doesn't correctly link targets for
# its Vorbis and FLAC dependencies, so they need to be manually found.
set(SNDFILE_DEP_LIBRARIES "")
set(SNDFILE_DEP_INCLUDE_DIRS "")
function(findAndCheckSndFileDependency DEP)
  find_package(${DEP} REQUIRED)
  if (${DEP}_FOUND)
    message(STATUS "Required SndFile dependency ${DEP} found.")
    set(SNDFILE_DEP_LIBRARIES ${SNDFILE_DEP_LIBRARIES} ${${DEP}_LIBRARIES})
    set(SNDFILE_DEP_INCLUDE_DIRS ${SNDFILE_DEP_INCLUDE_DIRS} ${${DEP}_INCLUDE_DIRS})
  else()
    message(FATAL_ERROR "Required SndFile dependency ${DEP} not found.")
  endif()
endfunction(findAndCheckSndFileDependency)
findAndCheckSndFileDependency(Ogg REQUIRED)
findAndCheckSndFileDependency(Vorbis REQUIRED)
findAndCheckSndFileDependency(VorbisEnc REQUIRED)
findAndCheckSndFileDependency(FLAC REQUIRED)
# Sndfile uses imported targets
find_package(SNDFILE REQUIRED)
if (SNDFILE_FOUND)
  message(STATUS "libsndfile found.")
else ()
  message(FATAL_ERROR "Required dependency libsndfile not found.")
endif ()

# ----------------------------- Lib -----------------------------

target_sources(
  flashlight-task-speech-recognition
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
  flashlight-task-speech-recognition
  PUBLIC
  ${SNDFILE_LIBRARIES}
  ${SNDFILE_DEP_LIBRARIES}
  )

target_include_directories(
  flashlight-task-speech-recognition
  PUBLIC
  ${SNDFILE_INCLUDE_DIRS}
  ${SNDFILE_DEP_INCLUDE_DIRS}
  )
