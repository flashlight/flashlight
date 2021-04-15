cmake_minimum_required(VERSION 3.10)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS fl-libraries)

build_test(SRC ${DIR}/audio/feature/CeplifterTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/audio/feature/DctTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/audio/feature/DerivativesTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/audio/feature/DitherTest.cpp LIBS ${LIBS})
build_test(
  SRC ${DIR}/audio/feature/MfccTest.cpp
  LIBS ${LIBS}
  PREPROC "FEATURE_TEST_DATADIR=\"${DIR}/audio/feature/data\""
  )
build_test(SRC ${DIR}/audio/feature/PreEmphasisTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/audio/feature/SpeechUtilsTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/audio/feature/TriFilterbankTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/audio/feature/WindowingTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/common/ProducerConsumerQueueTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/common/StringTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/common/SystemTest.cpp LIBS ${LIBS})
build_test(
  SRC ${DIR}/text/dictionary/DictionaryTest.cpp
  LIBS ${LIBS}
  PREPROC "DICTIONARY_TEST_DATADIR=\"${DIR}/text/dictionary\""
  )
build_test(
  SRC ${DIR}/text/tokenizer/TokenizerTest.cpp
  LIBS ${LIBS}
  PREPROC "TOKENIZER_TEST_DATADIR=\"${DIR}/text/tokenizer\""
  )
