cmake_minimum_required(VERSION 3.5.1)

# Download and build googletest
include(${CMAKE_MODULE_PATH}/BuildGoogleTest.cmake)

function(build_test SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
  add_dependencies(${target} gtest) # make sure gtest is built first
  target_link_libraries(
    ${target}
    PRIVATE
    task-asr
    ${GTEST_LIBRARIES}
    )
  target_include_directories(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/..
    ${GTEST_INCLUDE_DIR}
    )
  target_compile_definitions(
    ${target}
    PRIVATE
    "DATA_TEST_DATADIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data/testdata\""
    "DECODER_TEST_DATADIR=\"${CMAKE_CURRENT_SOURCE_DIR}/decoder/data\""
    )
  add_test(${target} ${target})
endfunction(build_test)

# Criterion
build_test(${CMAKE_CURRENT_SOURCE_DIR}/criterion/CriterionTest.cpp)
build_test(${CMAKE_CURRENT_SOURCE_DIR}/criterion/Seq2SeqTest.cpp)
build_test(${CMAKE_CURRENT_SOURCE_DIR}/criterion/attention/AttentionTest.cpp)
build_test(${CMAKE_CURRENT_SOURCE_DIR}/criterion/attention/WindowTest.cpp)
# Data
build_test(${CMAKE_CURRENT_SOURCE_DIR}/data/DataTest.cpp)
build_test(${CMAKE_CURRENT_SOURCE_DIR}/data/FeatureTest.cpp)
build_test(${CMAKE_CURRENT_SOURCE_DIR}/data/ListFileDatasetTest.cpp)
build_test(${CMAKE_CURRENT_SOURCE_DIR}/data/SoundTest.cpp)
# Decoder
build_test(${CMAKE_CURRENT_SOURCE_DIR}/decoder/ConvLmModuleTest.cpp)
build_test(${CMAKE_CURRENT_SOURCE_DIR}/decoder/DecoderTest.cpp)
# Runtime
build_test(${CMAKE_CURRENT_SOURCE_DIR}/runtime/RuntimeTest.cpp)
