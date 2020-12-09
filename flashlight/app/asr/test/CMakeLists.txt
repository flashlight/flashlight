cmake_minimum_required(VERSION 3.10)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS flashlight-app-asr)

# Criterion
build_test(${DIR}/criterion/CriterionTest.cpp ${LIBS} "" ON)
build_test(${DIR}/criterion/Seq2SeqTest.cpp ${LIBS} "" ON)
build_test(${DIR}/criterion/attention/AttentionTest.cpp ${LIBS} "" ON)
build_test(${DIR}/criterion/attention/WindowTest.cpp ${LIBS} "" ON)
# Data
build_test(
  ${DIR}/data/DataTest.cpp
  ${LIBS}
  "DATA_TEST_DATADIR=\"${DIR}/data/testdata\""
  ON
  )
build_test(${DIR}/data/FeatureTest.cpp ${LIBS} "" ON)
build_test(
  ${DIR}/data/ListFileDatasetTest.cpp
  ${LIBS}
  "DATA_TEST_DATADIR=\"${DIR}/data/testdata\""
  ON
  )
build_test(
  ${DIR}/data/SoundTest.cpp
  ${LIBS}
  "DATA_TEST_DATADIR=\"${DIR}/data/testdata\""
  ON
  )
# Decoder
build_test(
  ${DIR}/decoder/ConvLmModuleTest.cpp
  ${LIBS}
  "DECODER_TEST_DATADIR=\"${DIR}/decoder/data\""
  ON
  )
build_test(
  ${DIR}/decoder/DecoderTest.cpp
  ${LIBS}
  "DECODER_TEST_DATADIR=\"${DIR}/decoder/data\""
  ON
  )
# Runtime
build_test(${DIR}/runtime/RuntimeTest.cpp ${LIBS} "" ON)
# Augmentation
build_test(${DIR}/augmentation/SoundEffectTest.cpp ${LIBS} "" ON)
build_test(${DIR}/augmentation/SoundEffectConfigTest.cpp ${LIBS} "" ON)
