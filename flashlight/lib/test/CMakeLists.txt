cmake_minimum_required(VERSION 3.10)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS fl-libraries)

build_test(${DIR}/audio/feature/CeplifterTest.cpp ${LIBS} "" ON)
build_test(${DIR}/audio/feature/DctTest.cpp ${LIBS} "" ON)
build_test(${DIR}/audio/feature/DerivativesTest.cpp ${LIBS} "" ON)
build_test(${DIR}/audio/feature/DitherTest.cpp ${LIBS} "" ON)
build_test(
  ${DIR}/audio/feature/MfccTest.cpp
  ${LIBS}
  "FEATURE_TEST_DATADIR=\"${DIR}/audio/feature/data\""
  ON
  )
build_test(${DIR}/audio/feature/PreEmphasisTest.cpp ${LIBS} "" ON)
build_test(${DIR}/audio/feature/SpeechUtilsTest.cpp ${LIBS} "" ON)
build_test(${DIR}/audio/feature/TriFilterbankTest.cpp ${LIBS} "" ON)
build_test(${DIR}/audio/feature/WindowingTest.cpp ${LIBS} "" ON)
build_test(${DIR}/common/ProducerConsumerQueueTest.cpp ${LIBS} "" ON)
build_test(${DIR}/common/StringTest.cpp ${LIBS} "" ON)
build_test(${DIR}/common/SystemTest.cpp ${LIBS} "" ON)
build_test(
  ${DIR}/text/dictionary/DictionaryTest.cpp
  ${LIBS}
  "DICTIONARY_TEST_DATADIR=\"${DIR}/text/dictionary\""
  ON
  )
build_test(
  ${DIR}/text/tokenizer/TokenizerTest.cpp
  ${LIBS}
  "TOKENIZER_TEST_DATADIR=\"${DIR}/text/tokenizer\""
  ON
  )
