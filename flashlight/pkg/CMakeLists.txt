cmake_minimum_required(VERSION 3.10)

set(FL_PKG_VISION_DIR ${FL_PKG_DIR}/vision)
include(${FL_PKG_VISION_DIR}/CMakeLists.txt)

set(FL_PKG_RUNTIME_DIR ${FL_PKG_DIR}/runtime)
include(${FL_PKG_RUNTIME_DIR}/CMakeLists.txt)

# TODO find a nicer way of setting this instead of if/else
#cmake_dependent_option(FL_BUILD_ALL_APPS "Set all Flashlight apps ON by default" OFF "FL_BUILD_CORE;FL_BUILD_CONTRIB;" OFF)



#if (FL_BUILD_ALL_APPS)
  #cmake_dependent_option(FL_BUILD_APP_ASR "Build asr task for Flashlight" ON "FL_BUILD_CORE;FL_BUILD_CONTRIB;" OFF)
  #cmake_dependent_option(FL_BUILD_APP_IMGCLASS "Build image classification app" ON "FL_BUILD_CORE;FL_BUILD_CONTRIB;" OFF)
  #cmake_dependent_option(FL_BUILD_APP_OBJDET "Build object detection app" ON "FL_BUILD_CORE;FL_BUILD_CONTRIB;" OFF)
  #cmake_dependent_option(FL_BUILD_APP_LM "Build lm task for Flashlight" ON "FL_BUILD_CORE;FL_BUILD_CONTRIB;" OFF)
#else()
  #cmake_dependent_option(FL_BUILD_APP_ASR "Build asr task for Flashlight" OFF "FL_BUILD_CORE;FL_BUILD_CONTRIB;" OFF)
  #cmake_dependent_option(FL_BUILD_APP_IMGCLASS "Build image classification app" OFF "FL_BUILD_CORE;FL_BUILD_CONTRIB;" OFF)
  #cmake_dependent_option(FL_BUILD_APP_OBJDET "Build object detection app" OFF "FL_BUILD_CORE;FL_BUILD_CONTRIB;" OFF)
  #cmake_dependent_option(FL_BUILD_APP_LM "Build lm task for Flashlight" OFF "FL_BUILD_CORE;FL_BUILD_CONTRIB;" OFF)
#endif ()

#if (FL_BUILD_APP_ASR OR FL_BUILD_APP_IMGCLASS OR FL_BUILD_APP_LM OR FL_BUILD_APP_OBJDET)
  ## Find GLog
  #find_package(GLOG REQUIRED)
  #if (GLOG_FOUND)
    #message(STATUS "GLOG found")
    #if (FL_BUILD_STANDALONE)
      #setup_install_find_module(${CMAKE_MODULE_PATH}/FindGLOG.cmake)
    #endif()
  #else()
    #message(FATAL_ERROR "GLOG not found")
  #endif()

  ## Find gflags
  #find_package(gflags REQUIRED)
  #if (gflags_FOUND)
    #message(STATUS "gflags found")
    #if (FL_BUILD_STANDALONE)
      #setup_install_find_module(${CMAKE_MODULE_PATH}/Findgflags.cmake)
    #endif()
  #else()
    #message(FATAL_ERROR "gflags not found")
  #endif()

  #set(FL_APP_COMMON_ROOT_DIR ${FL_APPS_DIR}/common)
  #include(${FL_APP_COMMON_ROOT_DIR}/CMakeLists.txt)
  #set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} flashlight-app-common)
  #setup_install_headers(${FL_APP_COMMON_ROOT_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC}/app)
#endif ()

#if (FL_BUILD_APP_ASR)
  #message(STATUS "Building flashlight ASR app")
  #if (NOT FL_BUILD_DISTRIBUTED)
    #message(FATAL_ERROR "FL_BUILD_DISTRIBUTED must be enabled to build speech recognition")
  #endif()
  #set(FL_APP_ASR_ROOT_DIR ${FL_APPS_DIR}/asr)
  #include(${FL_APP_ASR_ROOT_DIR}/CMakeLists.txt)
  #set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} flashlight-app-asr)
  #setup_install_headers(${FL_APP_ASR_ROOT_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC}/app)
#endif ()

#if (FL_BUILD_APP_IMGCLASS OR FL_BUILD_APP_OBJDET) # imgclass is required to build objdet
  ## TODO(padentomasello): remove or better organize this code
  #if (NOT FL_BUILD_APP_IMGCLASS AND FL_BUILD_APP_OBJDET)
    #message(STATUS "FL_BUILD_APP_OBJDET is enabled - forcing FL_BUILD_APP_IMGCLASS to ON")
  #endif()
  #message(STATUS "Building flashlight Image Classification app")
  #if (NOT FL_BUILD_DISTRIBUTED)
    #message(FATAL_ERROR "FL_BUILD_DISTRIBUTED must be enabled for image classification")
  #endif()
  #set(FL_APP_IMGCLASS_ROOT_DIR ${FL_APPS_DIR}/imgclass)
  #include(${FL_APP_IMGCLASS_ROOT_DIR}/CMakeLists.txt)
  #set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} flashlight-app-imgclass)
  #setup_install_headers(${FL_APP_IMGCLASS_ROOT_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC}/app)
#endif ()

#if (FL_BUILD_APP_LM)
  #message(STATUS "Building flashlight LM app")
  #if (NOT FL_BUILD_DISTRIBUTED)
    #message(FATAL_ERROR "FL_BUILD_DISTRIBUTED must be enabled to language modeling")
  #endif()
  #set(FL_APP_LM_ROOT_DIR ${FL_APPS_DIR}/lm)
  #include(${FL_APP_LM_ROOT_DIR}/CMakeLists.txt)
  #set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} flashlight-app-lm)
  #setup_install_headers(${FL_APP_LM_ROOT_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC}/app)
#endif ()

#if (FL_BUILD_APP_OBJDET)
  #message(STATUS "Building flashlight Object Detection app")
  ## TODO (padentomasello) revist if we can pull out some functionality from IMGCLASS
  #if (NOT FL_BUILD_DISTRIBUTED)
    #message(FATAL_ERROR
    #"FL_BUILD_DISTRIBUTED must be enabled for object detection")
  #endif()
  #set(FL_APP_OBJDET_ROOT_DIR ${FL_APPS_DIR}/objdet)
  #include(${FL_APP_OBJDET_ROOT_DIR}/CMakeLists.txt)
  #set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} flashlight-app-objdet)
  #setup_install_headers(${FL_APP_OBJDET_ROOT_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC}/app)
#endif ()
