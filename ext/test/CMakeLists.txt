cmake_minimum_required(VERSION 3.5.1)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS flashlight)

build_test(
  ${DIR}/common/SequentialBuilderTest.cpp
  ${LIBS}
  "ARCHDIR=\"${DIR}/common/\""
  )

add_library(test_module_plugin MODULE
  ${DIR}/common/test_module_plugin.cpp)
target_include_directories(test_module_plugin
  PUBLIC "$<TARGET_PROPERTY:flashlight,INTERFACE_INCLUDE_DIRECTORIES>")
set_target_properties(test_module_plugin PROPERTIES
  POSITION_INDEPENDENT_CODE ON
      PREFIX "")
build_test(
  ${DIR}/common/ModulePluginTest.cpp
  ${LIBS}
  "PLUGINDIR=\"${CMAKE_CURRENT_BINARY_DIR}\""
  )
target_link_libraries(ModulePluginTest PUBLIC "-Wl,--export-dynamic")
add_dependencies(ModulePluginTest test_module_plugin)
