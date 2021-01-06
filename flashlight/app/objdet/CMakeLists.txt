cmake_minimum_required(VERSION 3.5.1)

add_library(
  flashlight-app-obj-det
  ""
  )


target_link_libraries(
  flashlight-app-obj-det
  PUBLIC
  ${gflags_LIBRARIES}
  ${GLOG_LIBRARIES}
  flashlight
  )

target_include_directories(
  flashlight-app-obj-det
  PUBLIC
  ${gflags_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  )

# Dataset
include(${CMAKE_CURRENT_LIST_DIR}/dataset/CMakeLists.txt)

# Criterion
include(${CMAKE_CURRENT_LIST_DIR}/criterion/CMakeLists.txt)

# NN
include(${CMAKE_CURRENT_LIST_DIR}/nn/CMakeLists.txt)

# TODO padentomasello have a "trainers" folder
add_executable(Detr ${CMAKE_CURRENT_LIST_DIR}/examples/Detr.cpp)
target_link_libraries(Detr flashlight-app-obj-det)

# Build tests
if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif ()
