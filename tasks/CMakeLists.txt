cmake_minimum_required(VERSION 3.5.1)

project(flashlight_tasks)

# ----------------------------- Setup -----------------------------
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# ------------------------- Build Tasks -------------------------

set(FLASHLIGHT_ASR_TASK_DIR "${PROJECT_SOURCE_DIR}/speech_recognition")

option(FL_BUILD_ASR_TASK "Build asr task for flashlight" ON)
if (FL_BUILD_ASR_TASK)
  add_subdirectory(${FLASHLIGHT_ASR_TASK_DIR})
endif ()
