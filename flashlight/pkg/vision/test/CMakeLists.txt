cmake_minimum_required(VERSION 3.10)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS flashlight-app-objdet)

build_test(SRC ${DIR}/criterion/SetCriterionTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/TransformerTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/TransformsTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/PositionalEmbeddingSineTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/criterion/HungarianTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/dataset/BoxUtilsTest.cpp LIBS ${LIBS})
