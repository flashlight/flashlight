cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ------------------------- Components -------------------------

# criterion-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/criterion)

# feature-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/feature)

# ------------------------- Library -------------------------

add_library(
  libraries-audio
  ""
  )

set_target_properties(
  libraries-audio
  PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 11
  )

target_link_libraries(
  libraries-audio
  PUBLIC
  libraries-audio-criterion
  libraries-audio-feature
  )

target_include_directories(
  libraries-audio
  PUBLIC
  ${PROJECT_SOURCE_DIR}
  )

target_compile_definitions(
  libraries-audio
  PUBLIC
  )
 
