cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- Optim -----------------------------
add_library(
  Optim
  INTERFACE
  )

set(
  OPTIM_SOURCES
  ${CMAKE_SOURCE_DIR}/flashlight/optim/Optimizers.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/optim/Utils.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/optim/AdamOptimizer.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/optim/AdadeltaOptimizer.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/optim/AdagradOptimizer.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/optim/AMSgradOptimizer.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/optim/NovogradOptimizer.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/optim/RMSPropOptimizer.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/optim/SGDOptimizer.cpp
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
