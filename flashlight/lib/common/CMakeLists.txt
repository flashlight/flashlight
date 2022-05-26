cmake_minimum_required(VERSION 3.10)

add_library(
  fl_lib_common
  ""
)

target_include_directories(
  fl_lib_common
  PRIVATE
  ${PROJECT_SOURCE_DIR}
  )

target_sources(
  fl_lib_common
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/System.cpp
  )

if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif ()

if (FL_CODE_COVERAGE)
  add_coverage_to_target(TARGET fl_lib_common)
endif()

# flashlight-text is required
if (NOT TARGET flashlight::flashlight-text)
  find_package(flashlight-text CONFIG)
  if (NOT TARGET flashlight::flashlight-text AND FL_BUILD_STANDALONE)
    message(STATUS "Flashlight Text not found - downloading and building from source")
    include(${CMAKE_MODULE_PATH}/BuildFlashlightText.cmake)
  elseif(TARGET flashlight::flashlight-text)
    message(STATUS "Found flashlight-text")
  else()
    message(FATAL_ERROR "Could not find requirement flashlight-text.")
  endif()
endif()

target_link_libraries(fl_lib_common PRIVATE flashlight::flashlight-text)
