cmake_minimum_required(VERSION 3.5.1)

set(DIR ${CMAKE_CURRENT_LIST_DIR})

# Add SoundTestUtil library
set(TEST_LIBS
    flashlight
    flashlight-app-asr
    flashlight-app-sfx)
build_test_library(${DIR}/SoundTestUtil.cpp "${TEST_LIBS}")

# Add all libs to be on the safe side
set(LIBS SoundTestUtil flashlight-app-asr flashlight-app-sfx)

build_test(
    ${DIR}/AdditveNoiseTest.cpp
    "${LIBS}"
    "DATA_TEST_DATADIR=\"${DIR}/data\"")
build_test(
    ${DIR}/ListFileTest.cpp
    "${LIBS}"
    "DATA_TEST_DATADIR=\"${DIR}/data\"")
build_test(
    ${DIR}/ReveberationTest.cpp
    "${LIBS}"
    "DATA_TEST_DATADIR=\"${DIR}/data\"")
build_test(
    ${DIR}/SoundEffectTest.cpp
    "${LIBS}"
    "DATA_TEST_DATADIR=\"${DIR}/data\"")
build_test(
    ${DIR}/SoundLoaderTest.cpp
    "${LIBS}"
    "DATA_TEST_DATADIR=\"${DIR}/data\"")
    build_test(
        ${DIR}/SoundEffectConfigFileTest.cpp
        "${LIBS}"
        "DATA_TEST_DATADIR=\"${DIR}/data\"")
