cmake_minimum_required(VERSION 3.16)

add_executable(
  benchmark
  ${CMAKE_CURRENT_LIST_DIR}/Run.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ModelBenchmarker.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
)

include(${CMAKE_CURRENT_LIST_DIR}/models/CMakeLists.txt)

target_link_libraries(
  benchmark
  flashlight
  fl_pkg_runtime
  fl_pkg_speech
  fl_pkg_vision
  fl_pkg_text
)

set_executable_output_directory(benchmark "${FL_BUILD_BINARY_OUTPUT_DIR}")
install(TARGETS benchmark RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
