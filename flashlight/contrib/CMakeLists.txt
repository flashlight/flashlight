cmake_minimum_required(VERSION 3.5.1)

# Root dir is `flashlight/contrib`
set(FL_CONTRIB_ROOT_PATH ${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/contrib)

add_library(
  Contrib
  INTERFACE
  )

# Include each submodule
include(${FL_CONTRIB_ROOT_PATH}/modules/CMakeLists.txt)

set(
  FLASHLIGHT_CONTRIB_LIBS
  ContribModules
  )

target_link_libraries(
  Contrib
  INTERFACE
  ${FLASHLIGHT_CONTRIB_LIBS}
  )
