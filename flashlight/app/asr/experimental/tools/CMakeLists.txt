cmake_minimum_required(VERSION 3.5.1)

function(build_tool SRCFILE)
  get_filename_component(target ${SRCFILE} NAME_WE)
  add_executable(${target} ${SRCFILE})
  target_link_libraries(
    ${target}
    PRIVATE
    wav2letter++
    )
  target_include_directories(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    )
endfunction(build_tool)

build_tool(${PROJECT_SOURCE_DIR}/tools/VoiceActivityDetection-CTC.cpp)
build_tool(${PROJECT_SOURCE_DIR}/tools/alignment/Align.cpp)
