cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ------------------------- Components -------------------------

# decoder-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/decoder)

# dictionary-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dictionary)

# tokenizer-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tokenizer)

# ------------------------- Library -------------------------

add_library(
  libraries-language
  ""
  )

set_target_properties(
  libraries-language
  PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 11
  )

target_link_libraries(
  libraries-language
  PUBLIC
  libraries-language-decoder
  libraries-language-dictionary
  libraries-language-tokenizer
  )

target_include_directories(
  libraries-language
  PUBLIC
  ${PROJECT_SOURCE_DIR}
  )

target_compile_definitions(
  libraries-language
  PUBLIC
  )
 
