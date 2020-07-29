cmake_minimum_required(VERSION 3.5.1)

project(flashlight_tasks)

# ----------------------------- Setup -----------------------------
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------- Configuration -----------------------------

option(W2L_BUILD_LIBRARIES_ONLY "Build wav2letter-libraries only" OFF)
option(W2L_BUILD_INFERENCE
  "Build inference even when W2L_BUILD_LIBRARIES_ONLY is set" OFF)
option(W2L_LIBRARIES_USE_CUDA "Use CUDA in libraries-only build" ON)
option(W2L_LIBRARIES_USE_KENLM "Use KenLM in libraries-only build" ON)
option(W2L_LIBRARIES_USE_MKL "Use MKL in libraries-only build" ON)
option(W2L_BUILD_FOR_PYTHON "Build Python bindings" OFF)
option(W2L_BUILD_TESTS "Build tests for wav2letter++" ON)
option(W2L_BUILD_EXAMPLES "Build examples for wav2letter++" ON)
option(W2L_BUILD_EXPERIMENTAL "Build internal experimental components for wav2letter++" OFF)
option(W2L_BUILD_SCRIPTS "Build internal scripts for wav2letter++" OFF)
option(W2L_BUILD_RECIPES "Build recipes" ON)
option(W2L_BUILD_TOOLS "Build audio tools" OFF)
set(KENLM_MAX_ORDER 6 CACHE STRING "Maximum ngram order for KenLM")

# ------------------------- Build Tasks -------------------------

set(FLASHLIGHT_ASR_TASK_DIR "${PROJECT_SOURCE_DIR}/speech_recognition")

option(FL_BUILD_ASR_TASK "Build asr task for flashlight" ON)
if (FL_BUILD_ASR_TASK)
  add_subdirectory(${FLASHLIGHT_ASR_TASK_DIR})
endif ()
