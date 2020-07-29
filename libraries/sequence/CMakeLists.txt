cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ------------------------- Components -------------------------

# criterion-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/criterion)

# ------------------------- Library -------------------------

add_library(
  libraries-sequence
  ""
  )

set_target_properties(
  libraries-sequence
  PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 11
  )

target_link_libraries(
  libraries-sequence
  PUBLIC
  libraries-sequence-criterion
  )

target_include_directories(
  libraries-sequence
  PUBLIC
  ${PROJECT_SOURCE_DIR}
  )

target_compile_definitions(
  libraries-sequence
  PUBLIC
  )
 
