cmake_minimum_required(VERSION 3.5.1)


# ------------------------ Global External Dependencies ------------------------

# flashlight
find_library(flashlight REQUIRED HINTS "${CMAKE_BINARY_DIR}/flashlight")


# ------------------------- Components -------------------------

# common-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/common)

# ------------------------- Library -------------------------

add_library(
  fl-extensions
  ""
  )

set_target_properties(
  fl-extensions
  PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 11
  )

target_link_libraries(
  fl-extensions
  PUBLIC
  libraries-common
  extension-common
  flashlight
  )

target_include_directories(
  fl-extensions
  PUBLIC
  ${PROJECT_SOURCE_DIR}
  )

target_compile_definitions(
  fl-extensions
  PUBLIC
  ${W2L_COMPILE_DEFINITIONS}
  )
 
 
