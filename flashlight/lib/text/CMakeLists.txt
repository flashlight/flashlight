cmake_minimum_required(VERSION 3.10)

add_library(
  fl_lib_text
  ""
  )

list(APPEND INSTALLABLE_TARGETS${INSTALLABLE_TARGETS} fl_lib_text)

target_include_directories(
  fl_lib_text
  PRIVATE
  ${PROJECT_SOURCE_DIR}
  )

# ------------------------- Components -------------------------

# decoder
include(${CMAKE_CURRENT_LIST_DIR}/decoder/CMakeLists.txt)

# dictionary
include(${CMAKE_CURRENT_LIST_DIR}/dictionary/CMakeLists.txt)

# tokenizer
include(${CMAKE_CURRENT_LIST_DIR}/tokenizer/CMakeLists.txt)

if(FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif()

if (FL_CODE_COVERAGE)
  add_coverage_to_target(TARGET fl_lib_text)
endif()
