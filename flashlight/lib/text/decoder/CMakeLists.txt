cmake_minimum_required(VERSION 3.10)

include(${CMAKE_CURRENT_LIST_DIR}/lm/CMakeLists.txt)

target_sources(
  fl_lib_text
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/LexiconDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/LexiconFreeDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/LexiconSeq2SeqDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/LexiconFreeSeq2SeqDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Trie.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  )
