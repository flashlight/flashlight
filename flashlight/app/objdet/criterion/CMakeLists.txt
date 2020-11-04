cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- NN Modules -----------------------------
target_sources(
  flashlight-app-obj-det
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Hungarian.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SetCriterion.cpp
  ${CMAKE_CURRENT_LIST_DIR}/HungarianLib.cpp
)
