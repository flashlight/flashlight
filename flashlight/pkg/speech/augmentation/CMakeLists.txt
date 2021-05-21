cmake_minimum_required(VERSION 3.13)

# ------------------------ libsox ------------------------
# Add the include dir and the libraries that are required for
# building libsox based sounds effects.
#
# tl;dr: When linking dynamically we only need to call find_package(sox).
# When linking statically, or when find_package(sox) fails, we download
# and build libsox and link statically with the libraries that we built.
# Details:
# Sox based sound augmentations require libsox and support for file
# formats like flac. On Ubuntu (and other Debian based) package managers
# there are two packages with these libraries. Package libsox has
# libsox and package libsox-fmt-base has the various sound file format
# support. When linking dynamically, we only need to point the linker
# to libsox.so which is in the libsox package. When linking statically,
# things are a bit more complicated. The libsox package has both the
# shared and the static libsox libraries. Libsox-fmt-base on the other
# hand has only shared libraries. Additionally, the static libsox is
# dependent on a list of other libraries. One way to write cmake for
# static linking with libsox is to write FindXXX.cmake modules to all
# these dependencies and hope to find them installed. The other way,
# and it is the one implemented here, is to download and bui  ld libsox
# locally.

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
  if (NOT SOX_FOUND AND FL_BUILD_STANDALONE)
      include(${CMAKE_MODULE_PATH}/Buildsox.cmake)
      add_dependencies(fl_pkg_speech sox)
  endif()

  if (NOT SOX_FOUND)
    message(
    FATAL_ERROR
    "FL_BUILD_APP_ASR_SFX_SOX is set but failed to find libsox"
    )
  endif()

  target_link_libraries(
    fl_pkg_speech
    PRIVATE
    ${sox_LIBRARIES}
    )

  target_include_directories(
    fl_pkg_speech
    PRIVATE
    ${sox_INCLUDE_DIRS}
    )

  target_sources(
    fl_pkg_speech
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/SoxWrapper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TimeStretch.cpp
  )

endif()
# ------------------------ sound effects ------------------------

target_sources(
  fl_pkg_speech
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
  fl_pkg_speech
  )
