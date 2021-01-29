cmake_minimum_required(VERSION 3.10)

target_sources(
  flashlight-app-asr
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/AdditiveNoise.cpp
  ${CMAKE_CURRENT_LIST_DIR}/GaussianNoise.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Reverberation.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SoundEffect.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SoundEffectConfig.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SoundEffectUtil.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ReverbAndNoise.cpp
  )


add_executable(
  fl_asr_sfx_apply
  ${CMAKE_CURRENT_LIST_DIR}/SoundEffectApply.cpp
  )

target_link_libraries(
  fl_asr_sfx_apply
  flashlight-app-asr
  )
