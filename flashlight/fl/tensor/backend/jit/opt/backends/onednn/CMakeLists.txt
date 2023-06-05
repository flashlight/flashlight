cmake_minimum_required(VERSION 3.10)

# ----------------------------- Sources -----------------------------

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnJitOptimizerExtension.cpp
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnOpFusion.cpp
)
