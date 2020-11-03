cmake_minimum_required(VERSION 3.5.1)

set(STB_INSTALL_PATH ${FL_INSTALL_INC_DIR}/stb)
message(STATUS "Will download stb from github")
include(${CMAKE_MODULE_PATH}/BuildStb.cmake)
add_dependencies(flashlight stb)
# Move stb headers at install time
install(DIRECTORY ${STB_SOURCE_DIR}
   DESTINATION ${STB_INSTALL_PATH}
   COMPONENT stb
   FILES_MATCHING
   PATTERN "*.hpp"
   PATTERN "*.h"
   PATTERN ".git" EXCLUDE
)

target_include_directories(flashlight PRIVATE ${stb_INCLUDE_DIRS})

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Transforms.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Jpeg.cpp
)
