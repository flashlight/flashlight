cmake_minimum_required(VERSION 3.13)

set(stb_INSTALL_PATH ${FL_INSTALL_INC_DIR}/stb)
find_package(stb)
if(NOT stb_FOUND)
  message(STATUS "Could not find stb_image.h. Will download stb from github")
  include(${CMAKE_MODULE_PATH}/BuildStb.cmake)
  add_dependencies(fl_pkg_vision stb)

  # Move stb headers at install time only if they weren't already found
  setup_install_headers(${stb_SOURCE_DIR} ${stb_INSTALL_PATH})
  setup_install_find_module(${CMAKE_MODULE_PATH}/Findstb.cmake)
else()
  message(STATUS "stb_image.h found: (include: ${stb_INCLUDE_DIRS})")
endif()

target_include_directories(fl_pkg_vision PRIVATE ${stb_INCLUDE_DIRS})

target_sources(
  fl_pkg_vision
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/BoxUtils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Coco.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CocoTransforms.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DistributedDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Imagenet.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Jpeg.cpp
  ${CMAKE_CURRENT_LIST_DIR}/LoaderDataset.h
  ${CMAKE_CURRENT_LIST_DIR}/Transforms.cpp
)
