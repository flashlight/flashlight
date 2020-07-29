cmake_minimum_required(VERSION 3.5.1)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lm)

add_library(
  libraries-language-decoder
  INTERFACE
  )

target_sources(
  libraries-language-decoder
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/LexiconDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LexiconFreeDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LexiconSeq2SeqDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LexiconFreeSeq2SeqDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Trie.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  )

target_link_libraries(
  libraries-language-decoder
  INTERFACE
  libraries-language-decoder-lm
  )
