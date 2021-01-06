cmake_minimum_required(VERSION 3.5.1)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS flashlight-app-obj-det)

build_test(SRC ${DIR}/criterion/SetCriterionTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/TransformerTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/TransformsTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/PositionalEmbeddingSineTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/criterion/HungarianTest.cpp LIBS ${LIBS})
#build_test(SRC ${DIR}/criterion/HungarianMatcherTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/dataset/BoxUtilsTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/PytorchTest.cpp LIBS ${LIBS})
