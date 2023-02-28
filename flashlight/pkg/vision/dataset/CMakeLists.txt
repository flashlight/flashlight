cmake_minimum_required(VERSION 3.16)

# stb_image.h
find_path(stb_INCLUDE_DIRS
  NAMES stb_image.h
  PATH_SUFFIXES include
  PATHS ${stb_BASE_DIR}
  ENV stb_BASE_DIR
  )
if (NOT stb_INCLUDE_DIRS)
  if (NOT FL_BUILD_STANDALONE)
    message(FATAL_ERROR "Required dependency stb not found - try setting stb_BASE_DIR")
  endif()
  message(STATUS "stb_image.h not found - will download from source")
  include(${PROJECT_SOURCE_DIR}/cmake/BuildStb.cmake)
endif()
message(STATUS "stb_image.h found: (include: ${stb_INCLUDE_DIRS})")
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
