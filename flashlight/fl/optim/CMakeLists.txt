cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- Optim -----------------------------

set(
  OPTIM_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/Optimizers.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/AdamOptimizer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/AdadeltaOptimizer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/AdagradOptimizer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/AMSgradOptimizer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/NAGOptimizer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/NovogradOptimizer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/RMSPropOptimizer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SGDOptimizer.cpp
  )

target_sources(
  flashlight
  PUBLIC
  $<BUILD_INTERFACE:${OPTIM_SOURCES}>
  )
