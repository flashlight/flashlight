cmake_minimum_required(VERSION 3.5.1)

# ------------------------- Library -------------------------

add_library(
  fl-extensions
  ""
  )

target_include_directories(
  fl-extensions
  PUBLIC
  ${PROJECT_SOURCE_DIR}
  )

target_compile_definitions(
  fl-extensions
  PUBLIC
  ${FL_LIBRARIES_COMPILE_DEFINITIONS}
  )

target_link_libraries(
  fl-extensions
  PUBLIC
  flashlight
  fl-libraries
  )

# ------------------------- Components -------------------------

# common
include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)
 
# --------------------------- Tests ---------------------------

# Build tests
option(FL_BUILD_TESTS "Build tests for flashlight" ON)
if (FL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests)
endif ()
