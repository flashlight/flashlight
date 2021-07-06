cmake_minimum_required(VERSION 3.10)

# ------------------------- Components -------------------------
add_library(
  fl_lib_audio
  ""
)

set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} fl_lib_audio)

target_include_directories(
  fl_lib_audio
  PRIVATE
  ${PROJECT_SOURCE_DIR}
  )

# feature-library
include(${CMAKE_CURRENT_LIST_DIR}/feature/CMakeLists.txt)
if(FL_BUILD_TESTS) 
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif()

if (FL_CODE_COVERAGE)
  add_coverage_to_target(TARGET fl_lib_audio)
endif()
