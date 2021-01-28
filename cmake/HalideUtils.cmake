# Adds a Halide library. Compiles a Halide AOT-generator at compile time,
# then runs the generator to produce header and lib artifacts that
# can be linked to a passed target.
#
# SRC - the src file for the project
# NAME - the name of the resulting target
# LIBS - libraries to which the generated library will be linked
# PREPROC - preprocessor defs to pass to the new target
# LINK_TO - target to which to link the generated pipeline
function(fl_add_and_link_halide_lib)
  set(options)
  set(oneValueArgs SRC NAME LINK_TO)
  set(multiValueArgs LIBS PREPROC)
  cmake_parse_arguments(fl_add_and_link_halide_lib
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN})

  # Generator binary
  set(GENERATOR_TARGET ${fl_add_and_link_halide_lib_NAME}_generator)
  # Generator output
  set(GENERATED_TARGET generate_${fl_add_and_link_halide_lib_NAME})
  add_executable(${GENERATOR_TARGET} ${fl_add_and_link_halide_lib_SRC})
  target_link_libraries(
    ${GENERATOR_TARGET}
    PRIVATE
    Halide::Halide
    ${LIBS})
  target_compile_definitions(
    ${GENERATOR_TARGET}
    PRIVATE
    ${PREPROC})
    
  # Run the generator
  # LLVM may leak memory during Halide compilation - if building with ASAN,
  # the generator might fail. Disable leack checking when executing generators
  set(GENERATED_LIB "${fl_add_and_link_halide_lib_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  set(GENERATED_HEADER "${fl_add_and_link_halide_lib_NAME}.h")
  add_custom_command(OUTPUT ${GENERATED_HEADER} "${GENERATED_LIB}"
    DEPENDS ${GENERATOR_TARGET}
    COMMAND ${CMAKE_COMMAND} -E env "ASAN_OPTIONS=detect_leaks=0" $<TARGET_FILE:${GENERATOR_TARGET}>
    VERBATIM)
  add_custom_target(${GENERATED_TARGET}
    DEPENDS ${GENERATED_HEADER} "${GENERATED_LIB}")
  add_dependencies(${GENERATED_TARGET} ${GENERATOR_TARGET})
  
  set(LIB_PATH ${CMAKE_CURRENT_BINARY_DIR}/${GENERATED_LIB})
  message(STATUS "Will generate AOT Halide Pipeline ${fl_add_and_link_halide_lib_NAME}")

  # TODO: use an IMPORTED target? Might be cleaner
  # add_library(${fl_add_and_link_halide_lib_NAME} STATIC IMPORTED)
  # set_target_properties(${fl_add_and_link_halide_lib_NAME} PROPERTIES
  #   INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}
  #   IMPORTED_LOCATION "${GENERATED_LIB}"
  #   INTERFACE_LINK_LIBRARIES Halide::Halide)
  # add_dependencies(${fl_add_and_link_halide_lib_NAME} ${GENERATED_TARGET})

  # Link the generated Halide lib to the target
  add_dependencies(${fl_add_and_link_halide_lib_LINK_TO} ${GENERATED_TARGET})
  # Ensure we can find generated headers
  target_include_directories(
    ${fl_add_and_link_halide_lib_LINK_TO} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
  # For now, this linkeage is private, which means the Flashlight core needs
  # to wrap Halide pipelines when exposing them to external binaries.
  # Properly installing the Halide lib will facilitate public linkeage.
  target_link_libraries(${fl_add_and_link_halide_lib_LINK_TO} PRIVATE ${LIB_PATH})
endfunction(fl_add_and_link_halide_lib)

# Adds a Halide library that is linked with Flashlight.
#
# If used from an included CMake list, we won't run into
# cmake_policy(SET CMP0079 NEW) issues. Halide pipelines
# compiled for tests should use fl_add_and_link_halide_lib
# instead since those are called via calls to `add_subdirectories`.
# CMake 3.13 resolves this.
function(fl_add_halide_lib)
    set(options)
  set(oneValueArgs SRC NAME)
  set(multiValueArgs LIBS PREPROC)
  cmake_parse_arguments(fl_add_halide_lib "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN})

  fl_add_and_link_halide_lib(
    SRC ${fl_add_halide_lib_SRC}
    NAME ${fl_add_halide_lib_NAME}
    LIBS ${fl_add_halide_lib_LIBS}
    PREPROC ${fl_add_halide_lib_PREPROC}
    LINK_TO flashlight
    )

  # TODO: An IMPORTED target could help with this
  # cmake_policy(SET CMP0079 NEW)
  # target_link_libraries(flashlight PUBLIC ...)
  # add_dependencies(flashlight ${GENERATED_TARGET})
  # Generated Halide libs get installed too
  # set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} ${LIB_PATH} PARENT_SCOPE)  
endfunction(fl_add_halide_lib)
