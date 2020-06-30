cmake_minimum_required(VERSION 3.5.1)

set(FL_LIBRARIES_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Download and build googletest
include(${FL_LIBRARIES_TEST_DIR}/cmake/BuildGoogleTest.cmake)

function(build_test SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
  if (TARGET gtest)
    add_dependencies(${target} gtest) # make sure gtest is built first
  endif()
  target_link_libraries(
    ${target}
    PRIVATE
    fl-libraries
    ${GTEST_LIBRARIES}
    )
  target_include_directories(
    ${target}
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${GTEST_INCLUDE_DIR}
    )
  target_compile_definitions(
    ${target}
    PRIVATE
    "FEATURE_TEST_DATADIR=\"${FL_LIBRARIES_TEST_DIR}/audio/feature/data\""
    "DICTIONARY_TEST_DATADIR=\"${FL_LIBRARIES_TEST_DIR}/language/dictionary\""
    )
  add_test(${target} ${target})
endfunction(build_test)

build_test(${FL_LIBRARIES_TEST_DIR}/audio/feature/CeplifterTest.cpp)
build_test(${FL_LIBRARIES_TEST_DIR}/audio/feature/DctTest.cpp)
build_test(${FL_LIBRARIES_TEST_DIR}/audio/feature/DerivativesTest.cpp)
build_test(${FL_LIBRARIES_TEST_DIR}/audio/feature/DitherTest.cpp)
build_test(${FL_LIBRARIES_TEST_DIR}/audio/feature/MfccTest.cpp)
build_test(${FL_LIBRARIES_TEST_DIR}/audio/feature/PreEmphasisTest.cpp)
build_test(${FL_LIBRARIES_TEST_DIR}/audio/feature/SpeechUtilsTest.cpp)
build_test(${FL_LIBRARIES_TEST_DIR}/audio/feature/TriFilterbankTest.cpp)
build_test(${FL_LIBRARIES_TEST_DIR}/audio/feature/WindowingTest.cpp)
build_test(${FL_LIBRARIES_TEST_DIR}/common/ProducerConsumerQueueTest.cpp)
build_test(${FL_LIBRARIES_TEST_DIR}/common/StringTest.cpp)
build_test(${FL_LIBRARIES_TEST_DIR}/common/SystemTest.cpp)
build_test(${FL_LIBRARIES_TEST_DIR}/language/dictionary/DictionaryTest.cpp)
