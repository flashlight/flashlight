cmake_minimum_required(VERSION 3.16)

add_library(fl_pkg_speech)

# Install lib and headers
list(APPEND INSTALLABLE_TARGETS fl_pkg_speech)
setup_install_headers(
  ${CMAKE_CURRENT_LIST_DIR}
  ${FL_INSTALL_INC_DIR_HEADER_LOC}/pkg)

# flashlight-text with flashlight-text-kenlm is required
if (NOT TARGET flashlight::flashlight-text)
  find_package(flashlight-text CONFIG 0.0.2)
  if (NOT (TARGET flashlight::flashlight-text AND TARGET flashlight::flashlight-text-kenlm) AND FL_BUILD_STANDALONE)
    message(STATUS "Flashlight Text not found - downloading and building from source")
    include(${PROJECT_SOURCE_DIR}/cmake/BuildFlashlightText.cmake)
  elseif(TARGET flashlight::flashlight-text)
    message(STATUS "Found flashlight-text")
  else()
    message(FATAL_ERROR "Could not find requirement flashlight-text.")
  endif()
endif()

# flashlight-sequence is required
if (NOT TARGET flashlight::flashlight-sequence)
  find_package(flashlight-sequence CONFIG)
  if (NOT TARGET flashlight::flashlight-sequence AND FL_BUILD_STANDALONE)
    message(STATUS "Flashlight Sequence not found - downloading and building from source")
    include(BuildFlashlightSequence)
  elseif(TARGET flashlight::flashlight-sequence)
    if (FL_USE_CUDA AND FL_SEQUENCE_USE_CUDA)
      message(STATUS "Found flashlight-sequence")
    else()
      message(FATAL_ERROR "Installed flashlight-sequence not built with required CUDA support")
    endif()
  else()
    message(FATAL_ERROR "Could not find requirement flashlight-sequence.")
  endif()
endif()

target_link_libraries(
  fl_pkg_speech
  PUBLIC
  flashlight
  fl_pkg_runtime
  flashlight::flashlight-text
  flashlight::flashlight-text-kenlm
  flashlight::flashlight-sequence
  )

# ------------------------ Components ------------------------

# Audio
include(${CMAKE_CURRENT_LIST_DIR}/audio/CMakeLists.txt)

# Common
include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)

# Criterion
include(${CMAKE_CURRENT_LIST_DIR}/criterion/CMakeLists.txt)

# Data
include(${CMAKE_CURRENT_LIST_DIR}/data/CMakeLists.txt)

# Decoder
include(${CMAKE_CURRENT_LIST_DIR}/decoder/CMakeLists.txt)

# Runtime
include(${CMAKE_CURRENT_LIST_DIR}/runtime/CMakeLists.txt)

# Augmentation
include(${CMAKE_CURRENT_LIST_DIR}/augmentation/CMakeLists.txt)

## --------------------------- Tests ---------------------------

# Build tests
if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif()

if (FL_CODE_COVERAGE)
  add_coverage_to_target(TARGET fl_pkg_speech)
endif()
