cmake_minimum_required(VERSION 3.10)

add_executable(fl_img_imagenet_resnet34
  ${CMAKE_CURRENT_LIST_DIR}/examples/Defines.cpp
  ${CMAKE_CURRENT_LIST_DIR}/examples/ImageNetResnet34.cpp)
target_link_libraries(
  fl_img_imagenet_resnet34
  fl_pkg_vision
  fl_pkg_runtime
  )
install(TARGETS fl_img_imagenet_resnet34 RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
set_executable_output_directory(fl_img_imagenet_resnet34 ${FL_BUILD_BINARY_OUTPUT_DIR}/imgclass)

add_executable(fl_img_imagenet_vit
  ${CMAKE_CURRENT_LIST_DIR}/examples/Defines.cpp
  ${CMAKE_CURRENT_LIST_DIR}/examples/ImageNetTransformer.cpp)
target_link_libraries(
  fl_img_imagenet_vit
  fl_pkg_vision
  fl_pkg_runtime
  )
install(TARGETS fl_img_imagenet_vit RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
set_executable_output_directory(fl_img_imagenet_vit ${FL_BUILD_BINARY_OUTPUT_DIR}/imgclass)

add_executable(fl_img_imagenet_eval
  ${CMAKE_CURRENT_LIST_DIR}/examples/Defines.cpp
  ${CMAKE_CURRENT_LIST_DIR}/examples/ImageNetEval.cpp)
target_link_libraries(
  fl_img_imagenet_eval
  fl_pkg_vision
  fl_pkg_runtime
  )
install(TARGETS fl_img_imagenet_eval RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
set_executable_output_directory(fl_img_imagenet_eval ${FL_BUILD_BINARY_OUTPUT_DIR}/imgclass)

if (FL_CODE_COVERAGE)
  add_coverage_to_target(TARGET fl_pkg_vision)
endif()
