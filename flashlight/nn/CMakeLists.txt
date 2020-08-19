cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- NN Modules -----------------------------
add_dependencies(flashlight cereal)

set(
  NN_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/Init.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp # utils
  ${CMAKE_CURRENT_LIST_DIR}/modules/Activations.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/AdaptiveSoftMax.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/BatchNorm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/Container.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/Conv2D.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/Dropout.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/Embedding.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/Identity.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/LayerNorm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/Linear.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/Loss.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/Module.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/Normalize.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/Padding.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/Pool2D.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/Reorder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/RNN.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/Transform.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/View.cpp
  ${CMAKE_CURRENT_LIST_DIR}/modules/WeightNorm.cpp
  )

if (FL_BUILD_DISTRIBUTED)
  set(
    NN_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/DistributedUtils.cpp
    ${NN_SOURCES}
  )
endif ()

target_sources(
  flashlight
  PUBLIC
  $<BUILD_INTERFACE:${NN_SOURCES}>
  )

if (TARGET CONAN_PKG::cereal)
  target_link_libraries(
    flashlight
    PUBLIC
    CONAN_PKG::cereal
    )
else()
  target_include_directories(
    flashlight
    PUBLIC
    ${cereal_INCLUDE_DIRS} # pulled in - see cmake/BuildCereal.cmake
    )
endif ()
