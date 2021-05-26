cmake_minimum_required(VERSION 3.10)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS fl_lib_common)

build_test(SRC ${DIR}/ProducerConsumerQueueTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/StringTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/SystemTest.cpp LIBS ${LIBS})
