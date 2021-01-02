cmake_minimum_required(VERSION 3.10)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS flashlight)

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

# Compile the generator binary
add_executable(test_halide_pipeline_generator ${DIR}/integrations/HalideTestPipeline.cpp)
target_link_libraries(test_halide_pipeline_generator Halide::Halide)
# Run the generator
# LLVM may leak memory during Halide compilation. If projects are built with address sanitizer enabled,
# this may cause generators to fail, making it hard to use Halide and address sanitizer at the same time.
# To work around this, we execute generators with an environment setting to disable leak checking.
set(FILTER_LIB "HalideTestPipeline${CMAKE_STATIC_LIBRARY_SUFFIX}")
add_custom_command(OUTPUT HalideTestPipeline.h "${FILTER_LIB}"
  DEPENDS test_halide_pipeline_generator
  COMMAND ${CMAKE_COMMAND} -E env "ASAN_OPTIONS=detect_leaks=0" $<TARGET_FILE:test_halide_pipeline_generator>
  VERBATIM)
add_custom_target(generate_test_halide_pipeline
  DEPENDS HalideTestPipeline.h "${FILTER_LIB}")
add_dependencies(generate_test_halide_pipeline test_halide_pipeline_generator)
# The test must have the generated output from the generator binary
# TODO: fixme -- actually link Halide
build_test(SRC ${DIR}/integrations/HalideTest.cpp LIBS ${LIBS} Halide::Halide "${CMAKE_CURRENT_BINARY_DIR}/${FILTER_LIB}")
add_dependencies(HalideTest generate_test_halide_pipeline)
target_include_directories(HalideTest PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
