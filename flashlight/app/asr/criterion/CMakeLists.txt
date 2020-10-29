cmake_minimum_required(VERSION 3.5.1)

# ---------------------------- Backend-specific -----------------------------
if (FL_BACKEND STREQUAL "CUDA")
  set(FLASHLIGHT_USE_CUDA ON)
endif ()

if (FLASHLIGHT_USE_CUDA)
  # Build WarpCTC
  set(WARPCTC_DIR ${FL_APP_ASR_ROOT_DIR}/third_party/warpctc)
  message(STATUS "Adding warpctc from dir: ${WARPCTC_DIR}")
  
  # HORRENDOUS HACK (remove me when we remove warpctc):
  # see https://cmake.org/Bug/view.php?id=14444
  add_subdirectory(${WARPCTC_DIR} ${PROJECT_BINARY_DIR}/third_party/warpctc)
  cuda_add_library(warpctc SHARED ${warpctc_SOURCES})
  target_link_libraries(warpctc ${warpctc_LIBS})
  set(WARPCTC_INCLUDE_DIR ${WARPCTC_DIR}/include)
  target_include_directories(warpctc PRIVATE ${WARPCTC_INCLUDE_DIR})

  target_sources(
    flashlight-app-asr
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/ConnectionistTemporalClassificationCriterion.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/CriterionUtils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/ForceAlignmentCriterion.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/FullConnectionCriterion.cpp
    )

  target_link_libraries(
    flashlight-app-asr
    PUBLIC
    ${CUDA_LIBRARIES}
    # Can remove once CTC constrained Viterbi is implemented in CUDA T62268085
    warpctc # added directly from subdir, please remove me
    )

  target_include_directories(
    flashlight-app-asr
    PUBLIC
    ${CUDA_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${WARPCTC_INCLUDE_DIR}> # I hope to remove me one day
    )
elseif (FLASHLIGHT_USE_CPU)
  target_sources(
    flashlight-app-asr
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/ConnectionistTemporalClassificationCriterion.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/CriterionUtils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/ForceAlignmentCriterion.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/FullConnectionCriterion.cpp
    )
else ()
  message(
    FATAL_ERROR
    "wav2letter criterions not implemented for ${FL_BACKEND} backend"
    )
endif ()

# ----------------------------- Criterion APIs -----------------------------

target_sources(
  flashlight-app-asr
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/ConnectionistTemporalClassificationCriterion.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CriterionUtils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ConnectionistTemporalClassificationCriterion.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CriterionUtils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ForceAlignmentCriterion.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Seq2SeqCriterion.cpp
  ${CMAKE_CURRENT_LIST_DIR}/FullConnectionCriterion.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TransformerCriterion.cpp
  # Attention
  ${CMAKE_CURRENT_LIST_DIR}/attention/ContentAttention.cpp
  ${CMAKE_CURRENT_LIST_DIR}/attention/LocationAttention.cpp
  ${CMAKE_CURRENT_LIST_DIR}/attention/MultiHeadAttention.cpp
  # Window
  ${CMAKE_CURRENT_LIST_DIR}/attention/MedianWindow.cpp
  ${CMAKE_CURRENT_LIST_DIR}/attention/SoftPretrainWindow.cpp
  ${CMAKE_CURRENT_LIST_DIR}/attention/SoftWindow.cpp
  ${CMAKE_CURRENT_LIST_DIR}/attention/StepWindow.cpp
  )
