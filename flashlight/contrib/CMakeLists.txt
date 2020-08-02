cmake_minimum_required(VERSION 3.5.1)

# Root dir is `flashlight/contrib`
set(FL_CONTRIB_ROOT_PATH ${CMAKE_CURRENT_LIST_DIR})

# Include each submodule
include(${FL_CONTRIB_ROOT_PATH}/modules/CMakeLists.txt)

set(
  FL_CONTRIB_SOURCES
  ${FL_CONTRIB_MODULE_SOURCES}
  )

target_sources(
  flashlight
  PUBLIC
  ${FL_CONTRIB_SOURCES}
  )
