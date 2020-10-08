cmake_minimum_required(VERSION 3.5.1)

add_library(
  flashlight-app-image-classification
  ""
  )

target_link_libraries(
  flashlight-app-image-classification
  PUBLIC
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  flashlight
  )

target_include_directories(
  flashlight-app-image-classification
  PUBLIC
  ${GFLAGS_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  )

include(${CMAKE_CURRENT_LIST_DIR}/dataset/CMakeLists.txt)


add_executable(Resnet34 ${CMAKE_CURRENT_LIST_DIR}/examples/Resnet34.cpp)
target_link_libraries(Resnet34 flashlight-app-image-classification)
