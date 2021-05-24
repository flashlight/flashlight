cmake_minimum_required(VERSION 3.10)

# Detect if we're doing an in-source or an out-of-source build
set(FL_PLUGIN_LINK_TARGET)
if (NOT TARGET flashlight)
  find_package(flashlight CONFIG REQUIRED)
  set(FL_PLUGIN_LINK_TARGET flashlight::flashlight)
else()
  set(FL_PLUGIN_LINK_TARGET flashlight)
endif()

function(compile_plugin)
  set(options)
  set(oneValueArgs SRC)
  set(multiValueArgs)
  cmake_parse_arguments(compile_plugin "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN})

  get_filename_component(src_name ${compile_plugin_SRC} NAME_WE)
  set(target "${src_name}")
  add_library(${target} MODULE ${compile_plugin_SRC})
  target_include_directories(
    ${target}
    PUBLIC
    "$<TARGET_PROPERTY:${FL_PLUGIN_LINK_TARGET},INTERFACE_INCLUDE_DIRECTORIES>"
    )
  set_target_properties(
    ${target}
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    )
  add_dependencies(${target} ${FL_PLUGIN_LINK_TARGET})
endfunction()

if (${FL_PLUGIN_MODULE_SRC_PATH} STREQUAL "")
  message(FATAL_ERROR
    "FL_PLUGIN_MODULE_SRC_PATH argument required. "
    "Usage: cmake -D FL_PLUGIN_MODULE_SRC_PATH=[path] [...]")
endif()

message(STATUS "Building plugin with src ${FL_PLUGIN_MODULE_SRC_PATH}")
compile_plugin(SRC ${FL_PLUGIN_MODULE_SRC_PATH})
