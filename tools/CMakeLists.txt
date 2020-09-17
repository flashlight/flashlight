cmake_minimum_required(VERSION 3.5.1)


option(FL_BUILD_TOOLS_IMAGE "Build image tools for for flashlight" ON)

if(FL_BUILD_TOOLS_IMAGE)
  set(FL_TOOLS_IMAGE_ROOT_DIR ${FLASHLIGHT_TOOLS_DIR}/image)
  include(${FL_TOOLS_IMAGE_ROOT_DIR}/CMakeLists.txt)
endif ()

