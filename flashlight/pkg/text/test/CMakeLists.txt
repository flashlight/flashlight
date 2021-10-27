cmake_minimum_required(VERSION 3.10)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS fl_pkg_text)

build_test(
  SRC ${DIR}/data/TextDatasetTest.cpp
  LIBS ${LIBS}
  PREPROC "TEXTDATASET_TEST_DATADIR=\"${DIR}/data/test_data\""
  )
