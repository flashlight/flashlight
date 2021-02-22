cmake_minimum_required(VERSION 3.10)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS flashlight-app-imgclass)

build_test(
  SRC ${DIR}/DistributedDatasetTest.cpp
  LIBS ${LIBS}
)
