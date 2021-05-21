cmake_minimum_required(VERSION 3.10)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS flashlight)

build_test(
  SRC ${DIR}/amp/DynamicScalerTest.cpp
  LIBS ${LIBS}
)

if(FL_BUILD_CONTRIB)
  build_test(
    SRC ${DIR}/common/SequentialBuilderTest.cpp
    LIBS ${LIBS}
    PREPROC "ARCHDIR=\"${DIR}/common/\""
  )
endif()

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

if (FL_EXT_BUILD_HALIDE)
  build_test(SRC ${DIR}/integrations/HalideTest.cpp LIBS ${LIBS})
  fl_add_and_link_halide_lib(
    SRC ${DIR}/integrations/HalideTestPipeline.cpp
    NAME HalideTestPipeline
    LINK_TO HalideTest)
endif()
