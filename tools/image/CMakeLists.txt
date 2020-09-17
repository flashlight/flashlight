cmake_minimum_required(VERSION 3.5.1)

add_library(
  flashlight-tools-image
  ""
)

add_library(
  flashlight-tools-image-af
  ""
)

add_library(
  flashlight-tools-image-fl
  ""
)

include(${CMAKE_CURRENT_LIST_DIR}/af/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/fl/CMakeLists.txt)

target_link_libraries(
  flashlight-tools-image-fl
  PUBLIC
  flashlight
)

target_link_libraries(
  flashlight-tools-image-af
  PUBLIC
  ArrayFire::${FL_AF_BACKEND}
)

target_link_libraries(
  flashlight-tools-image
  PUBLIC
  flashlight-tools-image-fl
  flashlight-tools-image-af
)
