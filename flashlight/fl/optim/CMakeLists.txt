cmake_minimum_required(VERSION 3.16)

target_sources(
  flashlight
  PRIVATE
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
