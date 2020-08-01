cmake_minimum_required(VERSION 3.5.1)

option(FL_BUILD_ASR_TASK "Build asr task for flashlight" ON)

if (FL_BUILD_ASR_TASK)
  set(FL_TASK_SPEECH_RECOGNITION_ROOT_DIR ${FLASHLIGHT_TASKS_DIR}/speech_recognition)
  include(${FL_TASK_SPEECH_RECOGNITION_ROOT_DIR}/CMakeLists.txt)
endif ()
