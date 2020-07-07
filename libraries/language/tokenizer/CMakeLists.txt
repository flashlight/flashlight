cmake_minimum_required(VERSION 3.5.1)

add_library(
  libraries-language-tokenizer
  INTERFACE
  )

target_sources(
  libraries-language-tokenizer
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Tokenizer.cpp
  )
