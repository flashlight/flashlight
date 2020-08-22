cmake_minimum_required(VERSION 3.5.1)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS fl-libraries)

build_test(${DIR}/audio/feature/CeplifterTest.cpp ${LIBS} "")
build_test(${DIR}/audio/feature/DctTest.cpp ${LIBS} "")
build_test(${DIR}/audio/feature/DerivativesTest.cpp ${LIBS} "")
build_test(${DIR}/audio/feature/DitherTest.cpp ${LIBS} "")
build_test(
  ${DIR}/audio/feature/MfccTest.cpp
  ${LIBS}
  "FEATURE_TEST_DATADIR=\"${DIR}/audio/feature/data\""
  )
build_test(${DIR}/audio/feature/PreEmphasisTest.cpp ${LIBS} "")
build_test(${DIR}/audio/feature/SpeechUtilsTest.cpp ${LIBS} "")
build_test(${DIR}/audio/feature/TriFilterbankTest.cpp ${LIBS} "")
build_test(${DIR}/audio/feature/WindowingTest.cpp ${LIBS} "")
build_test(${DIR}/common/ProducerConsumerQueueTest.cpp ${LIBS} "")
build_test(${DIR}/common/StringTest.cpp ${LIBS} "")
build_test(${DIR}/common/SystemTest.cpp ${LIBS} "")
build_test(
  ${DIR}/text/dictionary/DictionaryTest.cpp
  ${LIBS}
  "DICTIONARY_TEST_DATADIR=\"${DIR}/text/dictionary\""
  )
