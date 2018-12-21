cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- Optim -----------------------------
add_library(
  Optim
  INTERFACE
  )

set(
  OPTIM_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/optim/Optimizers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/optim/Utils.cpp
  )

target_sources(
  Optim
  INTERFACE
  $<BUILD_INTERFACE:${OPTIM_SOURCES}>
  )

target_link_libraries(
  Optim
  INTERFACE
  Common
)

target_include_directories(
  Optim
  INTERFACE
  ${ArrayFire_INCLUDE_DIRS}
  )
