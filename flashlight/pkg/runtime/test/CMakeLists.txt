cmake_minimum_required(VERSION 3.10)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS fl_pkg_runtime)

build_test(
  SRC ${DIR}/DynamicScalerTest.cpp
  LIBS ${LIBS}
)
