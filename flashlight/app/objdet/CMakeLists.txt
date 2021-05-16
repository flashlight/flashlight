cmake_minimum_required(VERSION 3.10)

add_executable(
  fl_img_imagenet_resnet50_backbone
  ${CMAKE_CURRENT_LIST_DIR}/examples/ImageNetResnet50Backbone.cpp)
target_link_libraries(
  fl_img_imagenet_resnet50_backbone
  fl_pkg_vision
  fl_pkg_runtime
  )
set_executable_output_directory(fl_img_imagenet_resnet50_backbone ${FL_BUILD_BINARY_OUTPUT_DIR}/objdet)

add_executable(fl_img_coco_detr ${CMAKE_CURRENT_LIST_DIR}/examples/CocoDetr.cpp)
target_link_libraries(
  fl_img_coco_detr 
  fl_pkg_vision
  fl_pkg_runtime
  )
set_executable_output_directory(fl_img_coco_detr ${FL_BUILD_BINARY_OUTPUT_DIR}/objdet)

# Build tests
if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif()
