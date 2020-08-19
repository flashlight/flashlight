cmake_minimum_required(VERSION 3.5.1)

# Root dir is `flashlight/contrib`
set(FL_CONTRIB_ROOT_PATH ${CMAKE_CURRENT_LIST_DIR})

# Include each submodule
include(${FL_CONTRIB_ROOT_PATH}/modules/CMakeLists.txt)

set(
  FLASHLIGHT_CONTRIB_SOURCES
  ${FLASHLIGHT_CONTRIB_MODULE_SOURCES}
  )

target_sources(
  flashlight
  PUBLIC
  ${FLASHLIGHT_CONTRIB_SOURCES}
  )
