cmake_minimum_required(VERSION 3.16)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
# TODO remove dependency on runtime components for these tests
set(LIBS fl_pkg_speech fl_pkg_runtime)

# Audio
build_test(SRC ${DIR}/audio/CeplifterTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/audio/DctTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/audio/DerivativesTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/audio/DitherTest.cpp LIBS ${LIBS})
build_test(
  SRC ${DIR}/audio/MfccTest.cpp
  LIBS ${LIBS}
  PREPROC "FEATURE_TEST_DATADIR=\"${DIR}/audio/data\""
  )
build_test(SRC ${DIR}/audio/PreEmphasisTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/audio/SpeechUtilsTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/audio/TriFilterbankTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/audio/WindowingTest.cpp LIBS ${LIBS})
# Criterion
build_test(SRC ${DIR}/criterion/CriterionTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/criterion/Seq2SeqTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/criterion/attention/AttentionTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/criterion/attention/WindowTest.cpp LIBS ${LIBS})
# Common
build_test(SRC ${DIR}/common/ProducerConsumerQueueTest.cpp LIBS ${LIBS})
# Data
build_test(SRC ${DIR}/data/FeaturizationTest.cpp LIBS ${LIBS})
build_test(
  SRC ${DIR}/data/ListFileDatasetTest.cpp
  LIBS ${LIBS}
  PREPROC "DATA_TEST_DATADIR=\"${DIR}/data/testdata\""
  )
build_test(
  SRC ${DIR}/data/SoundTest.cpp
  LIBS ${LIBS}
  PREPROC "DATA_TEST_DATADIR=\"${DIR}/data/testdata\""
  )
# Decoder
build_test(
  SRC ${DIR}/decoder/ConvLmModuleTest.cpp
  LIBS ${LIBS}
  PREPROC "DECODER_TEST_DATADIR=\"${DIR}/decoder/data\""
  )
# Runtime
build_test(SRC ${DIR}/runtime/RuntimeTest.cpp LIBS ${LIBS})
# Augmentation
build_test(SRC ${DIR}/augmentation/AdditiveNoiseTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/augmentation/GaussianNoiseTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/augmentation/SoundEffectTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/augmentation/SoundEffectConfigTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/augmentation/ReverberationTest.cpp LIBS ${LIBS})
if (FL_BUILD_APP_ASR_SFX_SOX)
  build_test(SRC ${DIR}/augmentation/TimeStretchTest.cpp LIBS ${LIBS})
endif()
