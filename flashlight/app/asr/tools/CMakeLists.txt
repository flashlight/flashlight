cmake_minimum_required(VERSION 3.10)

# TODO: move this to another place once we have other working tools.
# Use something similar to build_test where libs can be specified in a
# general way
function(build_tool SRCFILE target)
  add_executable(${target} ${SRCFILE})
  target_link_libraries(
    ${target}
    PRIVATE
    fl_pkg_speech
    )
  target_include_directories(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    )
  set_executable_output_directory(${target} "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")
  install(TARGETS ${target} RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
endfunction(build_tool)

build_tool(
  ${CMAKE_CURRENT_LIST_DIR}/VoiceActivityDetection-CTC.cpp
  fl_asr_voice_activity_detection_ctc
  )
build_tool(
  ${CMAKE_CURRENT_LIST_DIR}/alignment/Align.cpp
  fl_asr_align
  )
build_tool(
  ${CMAKE_CURRENT_LIST_DIR}/benchmark/ArchBenchmark.cpp
  fl_asr_arch_benchmark
  )
