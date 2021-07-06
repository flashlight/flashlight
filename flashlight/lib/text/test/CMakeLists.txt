cmake_minimum_required(VERSION 3.10)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS fl_lib_text fl_lib_common)

build_test(SRC ${DIR}/dictionary/DictionaryTest.cpp LIBS ${LIBS}
  PREPROC "DICTIONARY_TEST_DATADIR=\"${DIR}/dictionary\"")
build_test(SRC ${DIR}/tokenizer/TokenizerTest.cpp LIBS ${LIBS}
PREPROC "TOKENIZER_TEST_DATADIR=\"${DIR}/tokenizer\""
  )
