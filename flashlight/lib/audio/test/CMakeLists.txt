
cmake_minimum_required(VERSION 3.10)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS fl_lib_audio)

build_test(SRC ${DIR}/feature/CeplifterTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/feature/DctTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/feature/DerivativesTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/feature/DitherTest.cpp LIBS ${LIBS})
build_test(
  SRC ${DIR}/feature/MfccTest.cpp
  LIBS ${LIBS}
  PREPROC "FEATURE_TEST_DATADIR=\"${DIR}/audio/feature/data\""
  )
build_test(SRC ${DIR}/feature/PreEmphasisTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/feature/SpeechUtilsTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/feature/TriFilterbankTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/feature/WindowingTest.cpp LIBS ${LIBS})
