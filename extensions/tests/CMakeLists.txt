cmake_minimum_required(VERSION 3.5.1)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS flashlight)

build_test(
  ${DIR}/common/SequentialBuilderTest.cpp
  ${LIBS}
  "ARCHDIR=\"${DIR}/common/\""
  )
