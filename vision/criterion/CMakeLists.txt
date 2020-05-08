cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- NN Modules -----------------------------
add_library(
  VisionCriterion
  INTERFACE
  )

set(
  CRITERION_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/Hungarian.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SetCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HungarianLib.cpp
)


target_sources(
  VisionCriterion
  INTERFACE
  $<BUILD_INTERFACE:${CRITERION_SOURCES}>
  )

target_link_libraries(
  VisionCriterion
  INTERFACE
  Common
  NN
  )

target_include_directories(
  VisionCriterion
  INTERFACE
  ${ArrayFire_INCLUDE_DIRS}
  )
