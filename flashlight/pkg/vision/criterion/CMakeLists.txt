cmake_minimum_required(VERSION 3.16)

# ----------------------------- NN Modules -----------------------------
target_sources(
  fl_pkg_vision
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/SetCriterion.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Hungarian.cpp
  ${CMAKE_CURRENT_LIST_DIR}/HungarianImpl.cpp
)
