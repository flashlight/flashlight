cmake_minimum_required(VERSION 3.10)

# stb_image
set(stb_INSTALL_PATH ${FL_INSTALL_INC_DIR}/stb)
find_package(stb)
if(NOT stb_FOUND)
  message(STATUS "Could not find stb_image.h. Will download stb from github")
  include(${CMAKE_MODULE_PATH}/BuildStb.cmake)
  add_dependencies(flashlight stb)

  # Move stb headers at install time only if they weren't already found
  setup_install_headers(${stb_SOURCE_DIR} ${stb_INSTALL_PATH})
  setup_install_find_module(${CMAKE_MODULE_PATH}/Findstb.cmake)
else()
  message(STATUS "stb_image.h found: (include: ${stb_INCLUDE_DIRS})")
endif()
target_include_directories(flashlight PRIVATE ${stb_INCLUDE_DIRS})

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Transforms.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Jpeg.cpp
)
