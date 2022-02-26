cmake_minimum_required(VERSION 3.10)

# ----------------------------- Autograd -----------------------------
set(
  AUTOGRAD_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/Variable.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  )

target_sources(
  flashlight
  PRIVATE
  ${AUTOGRAD_SOURCES}
  )

include(${CMAKE_CURRENT_LIST_DIR}/tensor/CMakeLists.txt)
