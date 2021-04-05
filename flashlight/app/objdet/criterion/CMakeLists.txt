cmake_minimum_required(VERSION 3.10)

# ----------------------------- NN Modules -----------------------------
target_sources(
  flashlight-app-objdet
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/SetCriterion.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Hungarian.cpp
)
