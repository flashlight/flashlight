cmake_minimum_required(VERSION 3.16)

# ----------------------------- Sources -----------------------------

include(${CMAKE_CURRENT_LIST_DIR}/eval/CMakeLists.txt) # eval
include(${CMAKE_CURRENT_LIST_DIR}/ir/CMakeLists.txt)   # ir

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/JitBackend.cpp
  ${CMAKE_CURRENT_LIST_DIR}/JitTensorBase.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
)
