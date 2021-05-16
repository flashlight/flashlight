cmake_minimum_required(VERSION 3.10)

cmake_dependent_option(FL_BUILD_APP_ASR_TOOLS "Build ASR App tools" ON "FL_BUILD_APP_ASR" OFF)
cmake_dependent_option(
  FL_BUILD_APP_ASR_SFX_SOX
  "Build libsox-based sound effects augmentation" OFF
  "FL_BUILD_APP_ASR" OFF
  )
cmake_dependent_option(
  FL_USE_SOX_STATIC
  "Statically link libsox with sound effects" OFF
  "NOT BUILD_SHARED_LIBS" OFF
  )

add_library(
  flashlight-app-asr
  ""
  )

target_link_libraries(
  flashlight-app-asr
  PUBLIC
  ${GLOG_LIBRARIES}
  ${gflags_LIBRARIES}
  ${MKL_LIBRARIES}
  flashlight
  fl_pkg_runtime
  )

target_include_directories(
  flashlight-app-asr
  PUBLIC
  ${gflags_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${MKL_INCLUDE_DIR}
  )

# ------------------------ Components ------------------------

# Common
include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)

# Criterion
include(${CMAKE_CURRENT_LIST_DIR}/criterion/CMakeLists.txt)

# Data
include(${CMAKE_CURRENT_LIST_DIR}/data/CMakeLists.txt)

# Decoder
include(${CMAKE_CURRENT_LIST_DIR}/decoder/CMakeLists.txt)

# Runtime
include(${CMAKE_CURRENT_LIST_DIR}/runtime/CMakeLists.txt)

# Augmentation
include(${CMAKE_CURRENT_LIST_DIR}/augmentation/CMakeLists.txt)

# ----------------------------- Binaries -----------------------------
add_executable(fl_asr_train ${CMAKE_CURRENT_LIST_DIR}/Train.cpp)
add_executable(fl_asr_test ${CMAKE_CURRENT_LIST_DIR}/Test.cpp)
add_executable(fl_asr_decode ${CMAKE_CURRENT_LIST_DIR}/Decode.cpp)

target_link_libraries(fl_asr_train flashlight-app-asr ${CMAKE_DL_LIBS})
target_link_libraries(fl_asr_test flashlight-app-asr ${CMAKE_DL_LIBS})
target_link_libraries(fl_asr_decode flashlight-app-asr ${CMAKE_DL_LIBS})

set_executable_output_directory(fl_asr_train "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")
set_executable_output_directory(fl_asr_test "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")
set_executable_output_directory(fl_asr_decode "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")

install(TARGETS fl_asr_train RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
install(TARGETS fl_asr_test RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
install(TARGETS fl_asr_decode RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})

# ----------------------------- Tutorial Binaries -----------------------------

add_executable(fl_asr_tutorial_inference_ctc ${CMAKE_CURRENT_LIST_DIR}/tutorial/InferenceCTC.cpp)
add_executable(fl_asr_tutorial_finetune_ctc ${CMAKE_CURRENT_LIST_DIR}/tutorial/FinetuneCTC.cpp)

target_link_libraries(fl_asr_tutorial_inference_ctc flashlight-app-asr ${CMAKE_DL_LIBS})
target_link_libraries(fl_asr_tutorial_finetune_ctc flashlight-app-asr ${CMAKE_DL_LIBS})

set_executable_output_directory(fl_asr_tutorial_inference_ctc "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")
set_executable_output_directory(fl_asr_tutorial_finetune_ctc "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")

# --------------------------- Tools ---------------------------
if (FL_BUILD_APP_ASR_TOOLS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tools)
endif()

# --------------------------- Tests ---------------------------

# Build tests
if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif()

if (FL_CODE_COVERAGE)
  add_coverage_to_target(TARGET flashlight-app-asr)
endif()
