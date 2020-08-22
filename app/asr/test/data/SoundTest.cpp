/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>

#include <fstream>
#include <functional>
#include <sstream>

#include "flashlight/lib/common/System.h"
#include "flashlight/app/asr/data/Sound.h"

using namespace fl::lib;
using namespace fl::app::asr;

namespace {
std::string loadPath = "";

auto loadData = [](const std::string& filepath) {
  std::vector<double> data;
  std::ifstream file(filepath);
  std::istream_iterator<double> eos;
  std::istream_iterator<double> iit(file);
  std::copy(iit, eos, std::back_inserter(data));
  return data;
};
} // namespace

TEST(SoundTest, Mono) {
  auto audiopath =
      pathsConcat(loadPath, "test_mono.wav"); // 16-bit Signed Integer PCM
  auto datapath = pathsConcat(loadPath, "test_mono.dat");

  auto info = loadSoundInfo(audiopath);
  ASSERT_EQ(info.samplerate, 48000);
  ASSERT_EQ(info.channels, 1);
  ASSERT_EQ(info.frames, 24576);

  auto data = loadData(datapath);

  // Double
  auto vecDouble = loadSound<double>(audiopath);
  ASSERT_EQ(vecDouble.size(), info.channels * info.frames);
  for (int64_t i = 0; i < vecDouble.size(); ++i) {
    ASSERT_NEAR(vecDouble[i], data[i], 1E-8);
  }

  // Float
  auto vecFloat = loadSound<float>(audiopath);
  ASSERT_EQ(vecFloat.size(), info.channels * info.frames);

  for (int64_t i = 0; i < vecFloat.size(); ++i) {
    ASSERT_NEAR(vecFloat[i], data[i], 1E-6);
  }

  // scale by max value for short
  std::transform(
      data.begin(), data.end(), data.begin(), [](double d) -> double {
        return d * (1 << 15);
      });

  // Short
  auto vecShort = loadSound<short>(audiopath);
  ASSERT_EQ(vecShort.size(), info.channels * info.frames);

  for (int64_t i = 0; i < vecShort.size(); ++i) {
    ASSERT_NEAR(vecShort[i], data[i], 0.5);
  }

  // scale by (max value for int64_t / max value of short)
  std::transform(
      data.begin(), data.end(), data.begin(), [](double d) -> double {
        return d * (1 << 16);
      });
  // Int
  auto vecInt = loadSound<int>(audiopath);
  ASSERT_EQ(vecInt.size(), info.channels * info.frames);
  for (int64_t i = 0; i < vecInt.size(); ++i) {
    ASSERT_NEAR(vecInt[i], data[i], 25);
  }
}

TEST(SoundTest, Stereo) {
  auto audiopath =
      pathsConcat(loadPath, "test_stereo.wav"); // 16-bit Signed Integer PCM
  auto datapath = pathsConcat(loadPath, "test_stereo.dat");
  auto info = loadSoundInfo(audiopath);

  ASSERT_EQ(info.samplerate, 48000);
  ASSERT_EQ(info.channels, 2);
  ASSERT_EQ(info.frames, 24576);

  auto vecFloat = loadSound<float>(audiopath);
  ASSERT_EQ(vecFloat.size(), info.channels * info.frames);

  auto data = loadData(datapath);
  ASSERT_EQ(data.size(), info.channels * info.frames);

  for (int64_t i = 0; i < vecFloat.size(); ++i) {
    ASSERT_NEAR(vecFloat[i], data[i], 1E-6);
  }
}

TEST(SoundTest, OggReadWrite) {
  auto audiopath = pathsConcat(loadPath, "test_stereo.wav");
  auto outaudiopath = pathsConcat("/tmp", "test_stereo_out.ogg");
  auto oggaudiopath = pathsConcat(loadPath, "test_stereo.ogg");
  auto info = loadSoundInfo(audiopath);
  auto vecShort = loadSound<short>(audiopath);

  saveSound(
      outaudiopath,
      vecShort,
      info.samplerate,
      info.channels,
      SoundFormat::OGG,
      SoundSubFormat::VORBIS);
  auto vecFloatOut = loadSound<float>(outaudiopath);
  auto vecFloat = loadSound<float>(oggaudiopath);

  ASSERT_EQ(vecFloat.size(), vecFloatOut.size());

  for (int64_t i = 0; i < vecFloat.size(); ++i) {
    ASSERT_NEAR(vecFloat[i], vecFloatOut[i], 5E-3);
  }
}

TEST(SoundTest, StreamReadWrite) {
  auto audiopath = pathsConcat(loadPath, "test_stereo.wav");
  auto info = loadSoundInfo(audiopath);
  auto vecShort = loadSound<short>(audiopath);

  std::stringstream f;
  saveSound(
      f,
      vecShort,
      info.samplerate,
      info.channels,
      SoundFormat::WAV,
      SoundSubFormat::PCM_16);

  f.seekg(0);
  f.clear();
  auto infostream = loadSoundInfo(f);
  ASSERT_EQ(info.samplerate, infostream.samplerate);
  ASSERT_EQ(info.channels, infostream.channels);
  ASSERT_EQ(info.frames, infostream.frames);

  f.seekg(0);
  f.clear();
  auto vecShortStream = loadSound<short>(f);

  ASSERT_EQ(vecShort.size(), vecShortStream.size());
  for (int64_t i = 0; i < vecShort.size(); ++i) {
    ASSERT_EQ(vecShort[i], vecShortStream[i]);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

// Resolve directory for data
#ifdef DATA_TEST_DATADIR
  loadPath = DATA_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}
