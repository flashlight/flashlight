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


if (FL_EXT_BUILD_HALIDE)
  build_test(SRC ${DIR}/integrations/HalideTest.cpp LIBS ${LIBS})
  fl_add_and_link_halide_lib(
    SRC ${DIR}/integrations/HalideTestPipeline.cpp
    NAME HalideTestPipeline
    LINK_TO HalideTest)
endif()
