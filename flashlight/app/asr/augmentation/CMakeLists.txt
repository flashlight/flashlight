cmake_minimum_required(VERSION 3.10)

# ------------------------ libsox ------------------------
# Add the include dir and the libraries that are required for
# building libsox based sounds effects.

if (FL_BUILD_APP_ASR_SFX_SOX)
  # Define preprocessor macro FL_BUILD_APP_ASR_SFX_SOX
  target_compile_definitions(
    flashlight
    PUBLIC
    "-DFL_BUILD_APP_ASR_SFX_SOX"
    )

  # Find shared sox library
  if (NOT FL_USE_SOX_STATIC AND NOT FL_BUILD_STANDALONE)
      find_package(sox)
  endif()

  # Download and build static sox library.
  if (NOT SOX_FOUND)
      include(${CMAKE_MODULE_PATH}/Buildsox.cmake)
      add_dependencies(flashlight-app-asr sox)
  endif()

  target_link_libraries(
    flashlight-app-asr
    PRIVATE
    ${sox_LIBRARIES}
    )

  target_include_directories(
    flashlight-app-asr
    PRIVATE
    ${sox_INCLUDE_DIRS}
    )

  target_sources(
    flashlight-app-asr
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/SoxWrapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TimeStretch.cpp
  )

endif()
# ------------------------ sound effects ------------------------

target_sources(
  flashlight-app-asr
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/AdditiveNoise.cpp
  ${CMAKE_CURRENT_LIST_DIR}/GaussianNoise.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Reverberation.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SoundEffect.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SoundEffectConfig.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SoundEffectUtil.cpp
  )


add_executable(
  fl_asr_sfx_apply
  ${CMAKE_CURRENT_LIST_DIR}/SoundEffectApply.cpp
  )

set_executable_output_directory(
  fl_asr_sfx_apply
  "${FL_BUILD_BINARY_OUTPUT_DIR}/asr"
  )

target_link_libraries(
  fl_asr_sfx_apply
  flashlight-app-asr
  )
