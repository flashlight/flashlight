cmake_minimum_required(VERSION 3.16)

# ---------------------------- Backend-specific -----------------------------
if (FL_USE_CUDA)
  # Build WarpCTC
  set(WARPCTC_DIR ${FL_PKG_DIR}/speech/third_party/warpctc)
  message(STATUS "Adding warpctc from dir: ${WARPCTC_DIR}")

  add_subdirectory(${WARPCTC_DIR} ${PROJECT_BINARY_DIR}/third_party/warpctc)
  target_sources(
    fl_pkg_speech
    PRIVATE
    ${warpctc_SOURCES})
  # cuRAND is required and linked below
  set(WARPCTC_INCLUDE_DIR ${WARPCTC_DIR}/include)
  target_include_directories(fl_pkg_speech PRIVATE ${WARPCTC_INCLUDE_DIR})

  target_sources(
    fl_pkg_speech
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/ConnectionistTemporalClassificationCriterion.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/CriterionUtils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/ForceAlignmentCriterion.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/FullConnectionCriterion.cpp
    )

  target_link_libraries(
    fl_pkg_speech
    PRIVATE
    ${CUDA_LIBRARIES}
    )

  target_include_directories(
    fl_pkg_speech
    PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${WARPCTC_INCLUDE_DIR} # I hope to remove me one day
    )
else ()
  target_sources(
    fl_pkg_speech
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/ConnectionistTemporalClassificationCriterion.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/CriterionUtils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/ForceAlignmentCriterion.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/FullConnectionCriterion.cpp
    )
endif ()

# ----------------------------- Criterion APIs -----------------------------

target_sources(
  fl_pkg_speech
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
  ${CMAKE_CURRENT_LIST_DIR}/attention/Utils.cpp
  # Window
  ${CMAKE_CURRENT_LIST_DIR}/attention/MedianWindow.cpp
  ${CMAKE_CURRENT_LIST_DIR}/attention/SoftPretrainWindow.cpp
  ${CMAKE_CURRENT_LIST_DIR}/attention/SoftWindow.cpp
  ${CMAKE_CURRENT_LIST_DIR}/attention/StepWindow.cpp
  ${CMAKE_CURRENT_LIST_DIR}/attention/WindowBase.cpp
  )
