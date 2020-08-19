cmake_minimum_required(VERSION 3.5.1)

set(DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBS task-asr)

# Criterion
build_test(${DIR}/criterion/CriterionTest.cpp ${LIBS} "")
build_test(${DIR}/criterion/Seq2SeqTest.cpp ${LIBS} "")
build_test(${DIR}/criterion/attention/AttentionTest.cpp ${LIBS} "")
build_test(${DIR}/criterion/attention/WindowTest.cpp ${LIBS} "")
# Data
build_test(
  ${DIR}/data/DataTest.cpp
  ${LIBS}}
  "DATA_TEST_DATADIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data/testdata\""
  )
build_test(${DIR}/data/FeatureTest.cpp ${LIBS} "")
build_test(${DIR}/data/ListFileDatasetTest.cpp ${LIBS} "")
build_test(${DIR}/data/SoundTest.cpp ${LIBS} "")
# Decoder
build_test(${DIR}/decoder/ConvLmModuleTest.cpp ${LIBS} "")
build_test(
  ${DIR}/decoder/DecoderTest.cpp
  ${LIBS}
  "DECODER_TEST_DATADIR=\"${CMAKE_CURRENT_SOURCE_DIR}/decoder/data\""
  )
# Runtime
build_test(${DIR}/runtime/RuntimeTest.cpp ${LIBS} "")
