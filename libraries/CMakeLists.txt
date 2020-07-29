cmake_minimum_required(VERSION 3.5.1)

# ------------------------- Components -------------------------

# common-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/common)

# audio-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/audio)

# audio-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/sequence)

# decoder-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/language)

# ------------------------- Library -------------------------

add_library(
  fl-libraries
  ""
  )

set_target_properties(
  fl-libraries
  PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 11
  )

target_link_libraries(
  fl-libraries
  PUBLIC
  libraries-common
  libraries-audio
  libraries-sequence
  libraries-language
  )

target_include_directories(
  fl-libraries
  PUBLIC
  ${PROJECT_SOURCE_DIR}
  )

target_compile_definitions(
  fl-libraries
  PUBLIC
  )
 
# --------------------------- Tests ---------------------------

# Build tests
option(FL_BUILD_TESTS "Build tests for flashlight" ON)
if (FL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif ()
