cmake_minimum_required(VERSION 3.10)

target_sources(
  fl_lib_text
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/PartialFileReader.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Tokenizer.cpp
  )
