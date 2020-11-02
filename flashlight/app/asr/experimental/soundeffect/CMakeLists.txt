cmake_minimum_required(VERSION 3.5.1)

# ------------------------- Library -------------------------

add_library(
  flashlight-app-sfx
  ${CMAKE_CURRENT_LIST_DIR}/AdditiveNoise.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ListFile.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Reverberation.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Sound.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SoundEffect.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SoundLoader.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SoundEffectConfigFile.cpp
  )

target_link_libraries(
  flashlight-app-sfx
  PUBLIC
  flashlight
  flashlight-app-asr
  )

target_include_directories(
  flashlight-app-sfx
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/..
  )

# ----------------------------- Binaries -----------------------------
add_executable(ApplySoundEffect ${CMAKE_CURRENT_LIST_DIR}/ApplySoundEffect.cpp)

target_link_libraries(ApplySoundEffect flashlight-app-sfx)

# --------------------------- Tests ---------------------------

# Build tests
if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif ()
