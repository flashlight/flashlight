cmake_minimum_required(VERSION 3.5.1)

set(
  FL_CONTRIB_MODULE_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/AsymmetricConv1D.cpp
  ${CMAKE_CURRENT_LIST_DIR}/PositionEmbedding.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Residual.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SinusoidalPositionEmbedding.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Transformer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TDSBlock.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SpecAugment.cpp
  )
