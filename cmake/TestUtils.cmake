cmake_minimum_required(VERSION 3.5.1)

# Get Google Test
if (TARGET CONAN_PKG::gtest)
  message(STATUS "Found googletest installed by conan")
  # Use imported target
  set(GTEST_LIBRARIES CONAN_PKG::gtest)
else()
  find_package(GTest 1.10.0)
  if (NOT TARGET gtest AND NOT GTEST_FOUND)
    message(STATUS "googletest not found - will download and build from source")
    # Download and build googletest
    include(${CMAKE_MODULE_PATH}/BuildGoogleTest.cmake) # internally sets GTEST_LIBRARIES
  else()
    message(STATUS "gtest found: (include: ${GTEST_INCLUDE_DIRS}, lib: ${GTEST_BOTH_LIBRARIES}")
    set(GTEST_LIBRARIES ${GTEST_BOTH_LIBRARIES})
   endif()
endif()


function(build_test SRCFILE LINK_LIBRARIES PREPROC_DEFS)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
  if (TARGET gtest)
    add_dependencies(${target} gtest) # make sure gtest is built first
  endif()
  target_link_libraries(
    ${target}
    PUBLIC
    ${LINK_LIBRARIES}
    ${GTEST_LIBRARIES}
    )
  target_include_directories(
    ${target}
    PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GTEST_INCLUDE_DIRS}
    )
  target_compile_definitions(
    ${target}
    PUBLIC
    ${PREPROC_DEFS}
    )
  add_test(${target} ${target})
endfunction(build_test)
