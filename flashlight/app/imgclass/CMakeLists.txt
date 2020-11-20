cmake_minimum_required(VERSION 3.10)

add_library(
  flashlight-app-imgclass
  ""
  )

# Find gflags
find_package(gflags REQUIRED)
if (gflags_FOUND)
  message(STATUS "gflags found")
else()
  message(FATAL_ERROR "gflags not found")
endif()

target_link_libraries(
  flashlight-app-imgclass
  PUBLIC
  ${gflags_LIBRARIES}
  flashlight
  )

target_include_directories(
  flashlight-app-imgclass
  PUBLIC
  ${gflags_INCLUDE_DIRS}
  )

include(${CMAKE_CURRENT_LIST_DIR}/dataset/CMakeLists.txt)

add_executable(fl_imageNetResnet34
  ${CMAKE_CURRENT_LIST_DIR}/examples/ImageNetResnet34.cpp)
target_link_libraries(
  fl_imageNetResnet34
  flashlight-app-imgclass
  )
install(TARGETS fl_imageNetResnet34 RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
set_executable_output_directory(fl_imageNetResnet34 ${FL_BUILD_BINARY_OUTPUT_DIR}/imgclass)
