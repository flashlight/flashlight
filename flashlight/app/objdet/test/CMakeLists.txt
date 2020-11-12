cmake_minimum_required(VERSION 3.5.1)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS flashlight-app-obj-det)

build_test(${DIR}/criterion/SetCriterionTest.cpp ${LIBS} "")
build_test(${DIR}/TransformerTest.cpp ${LIBS} "")
build_test(${DIR}/TransformsTest.cpp ${LIBS} "")
build_test(${DIR}/PositionalEmbeddingSineTest.cpp ${LIBS} "")
build_test(${DIR}/criterion/HungarianTest.cpp ${LIBS} "")
build_test(${DIR}/criterion/HungarianMatcherTest.cpp ${LIBS} "")
build_test(${DIR}/dataset/BoxUtilsTest.cpp ${LIBS} "")
