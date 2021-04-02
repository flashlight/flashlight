cmake_minimum_required(VERSION 3.10)

add_library(
  flashlight-app-imgclass
  ""
  )

target_link_libraries(
  flashlight-app-imgclass
  PUBLIC
  ${GLOG_LIBRARIES}
  ${gflags_LIBRARIES}
  flashlight
  )

target_include_directories(
  flashlight-app-imgclass
  PUBLIC
  ${GLOG_INCLUDE_DIRS}
  ${gflags_INCLUDE_DIRS}
  )

include(${CMAKE_CURRENT_LIST_DIR}/dataset/CMakeLists.txt)

add_executable(fl_img_imagenet_resnet34
  ${CMAKE_CURRENT_LIST_DIR}/examples/Defines.cpp
  ${CMAKE_CURRENT_LIST_DIR}/examples/ImageNetResnet34.cpp)
target_link_libraries(
  fl_img_imagenet_resnet34
  flashlight-app-imgclass
  ${GLOG_LIBRARIES}
  )
install(TARGETS fl_img_imagenet_resnet34 RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
set_executable_output_directory(fl_img_imagenet_resnet34 ${FL_BUILD_BINARY_OUTPUT_DIR}/imgclass)

add_executable(fl_img_imagenet_vit
  ${CMAKE_CURRENT_LIST_DIR}/examples/Defines.cpp
  ${CMAKE_CURRENT_LIST_DIR}/examples/ImageNetTransformer.cpp)
target_link_libraries(
  fl_img_imagenet_vit
  flashlight-app-imgclass
  )
install(TARGETS fl_img_imagenet_vit RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
set_executable_output_directory(fl_img_imagenet_vit ${FL_BUILD_BINARY_OUTPUT_DIR}/imgclass)
