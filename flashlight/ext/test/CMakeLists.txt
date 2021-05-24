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

