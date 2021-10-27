cmake_minimum_required(VERSION 3.10)

add_library(
  flashlight-app-benchmark
  ""
  )

target_link_libraries(
  flashlight-app-benchmark
  PUBLIC
  ${GLOG_LIBRARIES}
  ${gflags_LIBRARIES}
  ${MKL_LIBRARIES}
  flashlight
  )

target_include_directories(
  flashlight-app-benchmark
  PUBLIC
  ${gflags_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${MKL_INCLUDE_DIR}
  )

target_sources(
  flashlight-app-benchmark
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/ModelBenchmarker.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  )

include(${CMAKE_CURRENT_LIST_DIR}/models/CMakeLists.txt)

add_executable(benchmark ${CMAKE_CURRENT_LIST_DIR}/Run.cpp)
target_link_libraries(
    benchmark
    flashlight-app-benchmark
    fl_pkg_speech
    fl_pkg_vision
    ${CMAKE_DL_LIBS})
set_executable_output_directory(benchmark "${FL_BUILD_BINARY_OUTPUT_DIR}")
install(TARGETS benchmark RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
