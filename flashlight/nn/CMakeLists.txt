cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- NN Modules -----------------------------
add_library(
  NN
  INTERFACE
  )

add_dependencies(NN cereal)

set(
  NN_SOURCES
  ${CMAKE_SOURCE_DIR}/flashlight/nn/Init.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/Utils.cpp # utils
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/Activations.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/AdaptiveSoftMax.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/BatchNorm.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/Container.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/Conv2D.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/Dropout.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/Embedding.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/LayerNorm.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/Linear.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/Loss.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/Module.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/Padding.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/Pool2D.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/Reorder.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/RNN.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/Transform.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/View.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/nn/modules/WeightNorm.cpp
  )

if (FL_BUILD_DISTRIBUTED)
  set(
    NN_SOURCES
    ${CMAKE_SOURCE_DIR}/flashlight/nn/DistributedUtils.cpp
    ${NN_SOURCES}
  )
endif ()

target_sources(
  NN
  INTERFACE
  $<BUILD_INTERFACE:${NN_SOURCES}>
  )

target_link_libraries(
  NN
  INTERFACE
  Autograd
  Common
  )

if (TARGET CONAN_PKG::cereal)
  target_link_libraries(
    NN
    INTERFACE
    CONAN_PKG::cereal
    )
else()
  target_include_directories(
    NN
    INTERFACE
    ${cereal_INCLUDE_DIRS} # pulled in - see cmake/BuildCereal.cmake
    )
endif ()
