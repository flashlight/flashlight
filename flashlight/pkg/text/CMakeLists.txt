cmake_minimum_required(VERSION 3.16)

add_library(fl_pkg_text)

# Install lib and headers
list(APPEND INSTALLABLE_TARGETS fl_pkg_text)
setup_install_headers(
  ${CMAKE_CURRENT_LIST_DIR}
  ${FL_INSTALL_INC_DIR_HEADER_LOC}/pkg)

# flashlight-text is required
if (NOT TARGET flashlight::flashlight-text)
  find_package(flashlight-text CONFIG 0.0.2)
  if (NOT TARGET flashlight::flashlight-text AND FL_BUILD_STANDALONE)
    message(STATUS "Flashlight Text not found - downloading and building from source")
    include(${PROJECT_SOURCE_DIR}/cmake/BuildFlashlightText.cmake)
  elseif(TARGET flashlight::flashlight-text)
    message(STATUS "Found flashlight-text")
  else()
    message(FATAL_ERROR "Could not find requirement flashlight-text.")
  endif()
endif()

target_link_libraries(
  fl_pkg_text
  PUBLIC
  flashlight
  flashlight::flashlight-text
  )
# ------------------------ Components ------------------------

# Data
include(${CMAKE_CURRENT_LIST_DIR}/data/CMakeLists.txt)

## --------------------------- Tests ---------------------------

## Build tests
if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif()

if (FL_CODE_COVERAGE)
  add_coverage_to_target(TARGET fl_pkg_text)
endif()
