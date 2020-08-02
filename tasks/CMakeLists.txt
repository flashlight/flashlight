cmake_minimum_required(VERSION 3.5.1)

option(FL_BUILD_TASK_SPEECH_RECOGNITION "Build asr task for flashlight" ON)

if (FL_BUILD_TASK_SPEECH_RECOGNITION)
  if (NOT FL_BUILD_DISTRIBUTED)
    message(FATAL_ERROR "FL_BUILD_DISTRIBUTED must be enabled to build speech recognition")
  endif()
  set(FL_TASK_SPEECH_RECOGNITION_ROOT_DIR ${FLASHLIGHT_TASKS_DIR}/speech_recognition)
  include(${FL_TASK_SPEECH_RECOGNITION_ROOT_DIR}/CMakeLists.txt)
endif ()
