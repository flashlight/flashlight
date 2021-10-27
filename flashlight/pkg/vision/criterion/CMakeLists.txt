cmake_minimum_required(VERSION 3.10)

# ----------------------------- NN Modules -----------------------------
target_sources(
  fl_pkg_vision
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/SetCriterion.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Hungarian.cpp
)

target_link_libraries(
  fl_pkg_vision
  PRIVATE
  fl_lib_set
)
