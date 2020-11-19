cmake_minimum_required(VERSION 3.10)

add_library(
  flashlight-app-imclass
  ""
  )

# Find GFlags
find_package(GFLAGS REQUIRED)
if (GFLAGS_FOUND)
  message(STATUS "GFLAGS found")
else()
  message(FATAL_ERROR "GFLAGS not found")
endif()

target_link_libraries(
  flashlight-app-imclass
  PUBLIC
  ${gflags_LIBRARIES}
  flashlight
  )

target_include_directories(
  flashlight-app-imclass
  PUBLIC
  ${gflags_INCLUDE_DIRS}
  )

include(${CMAKE_CURRENT_LIST_DIR}/dataset/CMakeLists.txt)

add_executable(imageNetResnet34
  ${CMAKE_CURRENT_LIST_DIR}/examples/ImageNetResnet34.cpp)
target_link_libraries(
  imageNetResnet34
  flashlight-app-imclass
  )
