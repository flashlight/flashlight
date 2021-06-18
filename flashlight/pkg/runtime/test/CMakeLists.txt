cmake_minimum_required(VERSION 3.10)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS fl_pkg_runtime)

build_test(
  SRC ${DIR}/DynamicScalerTest.cpp
  LIBS ${LIBS}
)

add_library(test_module_plugin MODULE
  ${DIR}/plugin/test_module_plugin.cpp)
target_include_directories(test_module_plugin
  PUBLIC "$<TARGET_PROPERTY:flashlight,INTERFACE_INCLUDE_DIRECTORIES>")
set_target_properties(test_module_plugin PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  PREFIX "")
add_dependencies(test_module_plugin flashlight)

build_test(
  SRC ${DIR}/plugin/ModulePluginTest.cpp
  LIBS ${LIBS}
  PREPROC "PLUGINDIR=\"${CMAKE_CURRENT_BINARY_DIR}\""
  )
add_dependencies(ModulePluginTest test_module_plugin)
