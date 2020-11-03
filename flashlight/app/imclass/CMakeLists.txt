cmake_minimum_required(VERSION 3.5.1)

add_library(
  flashlight-app-imclass
  ""
  )

target_link_libraries(
  flashlight-app-imclass
  PUBLIC
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  flashlight
  )

target_include_directories(
  flashlight-app-imclass
  PUBLIC
  ${GFLAGS_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  )

include(${CMAKE_CURRENT_LIST_DIR}/dataset/CMakeLists.txt)


add_executable(imageNetResnet34
${CMAKE_CURRENT_LIST_DIR}/examples/ImageNetResnet34.cpp)
target_link_libraries(imageNetResnet34 flashlight-app-imclass)
