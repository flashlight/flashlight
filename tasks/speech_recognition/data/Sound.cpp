/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Sound.h"

#include <fstream>
#include <string>
#include <unordered_map>

#include <sndfile.h>

using namespace fl::task::asr;

namespace {

struct EnumClassHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

const std::unordered_map<SoundFormat, int, EnumClassHash> formats{
    {SoundFormat::WAV, SF_FORMAT_WAV},
    {SoundFormat::AIFF, SF_FORMAT_AIFF},
    {SoundFormat::AU, SF_FORMAT_AU},
    {SoundFormat::RAW, SF_FORMAT_RAW},
    {SoundFormat::PAF, SF_FORMAT_PAF},
    {SoundFormat::SVX, SF_FORMAT_SVX},
    {SoundFormat::NIST, SF_FORMAT_NIST},
    {SoundFormat::VOC, SF_FORMAT_VOC},
    {SoundFormat::IRCAM, SF_FORMAT_IRCAM},
    {SoundFormat::W64, SF_FORMAT_W64},
    {SoundFormat::MAT4, SF_FORMAT_MAT4},
    {SoundFormat::MAT5, SF_FORMAT_MAT5},
    {SoundFormat::PVF, SF_FORMAT_PVF},
    {SoundFormat::XI, SF_FORMAT_XI},
    {SoundFormat::HTK, SF_FORMAT_HTK},
    {SoundFormat::SDS, SF_FORMAT_SDS},
    {SoundFormat::AVR, SF_FORMAT_AVR},
    {SoundFormat::WAVEX, SF_FORMAT_WAVEX},
    {SoundFormat::SD2, SF_FORMAT_SD2},
    {SoundFormat::FLAC, SF_FORMAT_FLAC},
    {SoundFormat::CAF, SF_FORMAT_CAF},
    {SoundFormat::WVE, SF_FORMAT_WVE},
    {SoundFormat::OGG, SF_FORMAT_OGG},
    {SoundFormat::MPC2K, SF_FORMAT_MPC2K},
    {SoundFormat::RF64, SF_FORMAT_RF64}};

const std::unordered_map<SoundSubFormat, int, EnumClassHash> subformats{
    {SoundSubFormat::PCM_S8, SF_FORMAT_PCM_S8},
    {SoundSubFormat::PCM_16, SF_FORMAT_PCM_16},
    {SoundSubFormat::PCM_24, SF_FORMAT_PCM_24},
    {SoundSubFormat::PCM_32, SF_FORMAT_PCM_32},
    {SoundSubFormat::PCM_U8, SF_FORMAT_PCM_U8},
    {SoundSubFormat::FLOAT, SF_FORMAT_FLOAT},
    {SoundSubFormat::DOUBLE, SF_FORMAT_DOUBLE},
    {SoundSubFormat::ULAW, SF_FORMAT_ULAW},
    {SoundSubFormat::ALAW, SF_FORMAT_ALAW},
    {SoundSubFormat::IMA_ADPCM, SF_FORMAT_IMA_ADPCM},
    {SoundSubFormat::MS_ADPCM, SF_FORMAT_MS_ADPCM},
    {SoundSubFormat::GSM610, SF_FORMAT_GSM610},
    {SoundSubFormat::VOX_ADPCM, SF_FORMAT_VOX_ADPCM},
    {SoundSubFormat::G721_32, SF_FORMAT_G721_32},
    {SoundSubFormat::G723_24, SF_FORMAT_G723_24},
    {SoundSubFormat::G723_40, SF_FORMAT_G723_40},
    {SoundSubFormat::DWVW_12, SF_FORMAT_DWVW_12},
    {SoundSubFormat::DWVW_16, SF_FORMAT_DWVW_16},
    {SoundSubFormat::DWVW_24, SF_FORMAT_DWVW_24},
    {SoundSubFormat::DWVW_N, SF_FORMAT_DWVW_N},
    {SoundSubFormat::DPCM_8, SF_FORMAT_DPCM_8},
    {SoundSubFormat::DPCM_16, SF_FORMAT_DPCM_16},
    {SoundSubFormat::VORBIS, SF_FORMAT_VORBIS}};
} // namespace

namespace fl {
namespace task {
namespace asr {

extern "C" {

static sf_count_t sf_vio_ro_get_filelen(void* user_data) {
  std::istream* f = reinterpret_cast<std::istream*>(user_data);
  auto pos = f->tellg();
  f->seekg(0, std::ios_base::end);
  auto size = f->tellg();
  f->seekg(pos, std::ios_base::beg);
  return (sf_count_t)size;
}

static sf_count_t
sf_vio_ro_seek(sf_count_t offset, int whence, void* user_data) {
  std::istream* f = reinterpret_cast<std::istream*>(user_data);
  std::ios_base::seekdir way;
  switch (whence) {
    case SEEK_CUR:
      way = std::ios_base::cur;
      break;
    case SEEK_SET:
      way = std::ios_base::beg;
      break;
    case SEEK_END:
      way = std::ios_base::end;
      break;
    default:
      throw std::invalid_argument("whence is invalid");
  }
  f->seekg(offset, way);
  return offset;
}

static sf_count_t sf_vio_ro_read(void* ptr, sf_count_t count, void* user_data) {
  std::istream* f = reinterpret_cast<std::istream*>(user_data);
  f->read((char*)ptr, count);
  auto n = f->gcount();
  if (!f->good()) {
    f->clear();
  }
  return n;
}

static sf_count_t sf_vio_ro_write(
    const void* /* ptr */,
    sf_count_t /* count */,
    void* /* user_data */) {
  throw std::invalid_argument("read-only stream");
  return 0;
}

static sf_count_t sf_vio_ro_tell(void* user_data) {
  std::istream* f = reinterpret_cast<std::istream*>(user_data);
  return f->tellg();
}

static sf_count_t sf_vio_wo_get_filelen(void* user_data) {
  std::ostream* f = reinterpret_cast<std::ostream*>(user_data);
  auto pos = f->tellp();
  f->seekp(0, std::ios_base::end);
  auto size = f->tellp();
  f->seekp(pos, std::ios_base::beg);
  return (sf_count_t)size;
}

static sf_count_t
sf_vio_wo_seek(sf_count_t offset, int whence, void* user_data) {
  std::ostream* f = reinterpret_cast<std::ostream*>(user_data);
  std::ios_base::seekdir way;
  switch (whence) {
    case SEEK_CUR:
      way = std::ios_base::cur;
      break;
    case SEEK_SET:
      way = std::ios_base::beg;
      break;
    case SEEK_END:
      way = std::ios_base::end;
      break;
    default:
      throw std::invalid_argument("whence is invalid");
  }
  f->seekp(offset, way);
  return offset;
}

static sf_count_t
sf_vio_wo_read(void* /* ptr */, sf_count_t /* count */, void* /* user_data */) {
  throw std::invalid_argument("write-only stream");
  return 0;
}

static sf_count_t
sf_vio_wo_write(const void* ptr, sf_count_t count, void* user_data) {
  std::ostream* f = reinterpret_cast<std::ostream*>(user_data);
  auto pos = f->tellp();
  f->write((const char*)ptr, count);
  return f->tellp() - pos;
}

static sf_count_t sf_vio_wo_tell(void* user_data) {
  std::ostream* f = reinterpret_cast<std::ostream*>(user_data);
  return f->tellp();
}

} /* extern "C" */

SoundInfo loadSoundInfo(const std::string& filename) {
  std::ifstream f(filename);
  if (!f.is_open()) {
    throw std::runtime_error("could not open file for read " + filename);
  }
  return loadSoundInfo(f);
}

SoundInfo loadSoundInfo(std::istream& f) {
  SF_VIRTUAL_IO vsf = {sf_vio_ro_get_filelen,
                       sf_vio_ro_seek,
                       sf_vio_ro_read,
                       sf_vio_ro_write,
                       sf_vio_ro_tell};

  SNDFILE* file;
  SF_INFO info;

  /* mandatory */
  info.format = 0;

  if (!(file = sf_open_virtual(&vsf, SFM_READ, &info, &f))) {
    throw std::runtime_error(
        "loadSoundInfo: unknown format or could not open stream");
  }

  sf_close(file);

  SoundInfo usrinfo;
  usrinfo.frames = info.frames;
  usrinfo.samplerate = info.samplerate;
  usrinfo.channels = info.channels;
  return usrinfo;
}

template <typename T>
std::vector<T> loadSound(const std::string& filename) {
  std::ifstream f(filename);
  if (!f.is_open()) {
    throw std::runtime_error("could not open file " + filename);
  }
  return loadSound<T>(f);
}

template <typename T>
std::vector<T> loadSound(std::istream& f) {
  SF_VIRTUAL_IO vsf = {sf_vio_ro_get_filelen,
                       sf_vio_ro_seek,
                       sf_vio_ro_read,
                       sf_vio_ro_write,
                       sf_vio_ro_tell};
  SNDFILE* file;
  SF_INFO info;

  info.format = 0;

  if (!(file = sf_open_virtual(&vsf, SFM_READ, &info, &f))) {
    throw std::runtime_error(
        "loadSound: unknown format or could not open stream");
  }

  std::vector<T> in(info.frames * info.channels);
  sf_count_t nframe;
  if (std::is_same<T, float>::value) {
    nframe =
        sf_readf_float(file, reinterpret_cast<float*>(in.data()), info.frames);
  } else if (std::is_same<T, double>::value) {
    nframe = sf_readf_double(
        file, reinterpret_cast<double*>(in.data()), info.frames);
  } else if (std::is_same<T, int>::value) {
    nframe = sf_readf_int(file, reinterpret_cast<int*>(in.data()), info.frames);
  } else if (std::is_same<T, short>::value) {
    nframe =
        sf_readf_short(file, reinterpret_cast<short*>(in.data()), info.frames);
  } else {
    throw std::logic_error("loadSound: called with unsupported T");
  }
  sf_close(file);
  if (nframe != info.frames) {
    throw std::runtime_error("loadSound: read error");
  }
  return in;
}

template <typename T>
void saveSound(
    const std::string& filename,
    const std::vector<T>& input,
    int64_t samplerate,
    int64_t channels,
    SoundFormat format,
    SoundSubFormat subformat) {
  std::ofstream f(filename);
  if (!f.is_open()) {
    throw std::runtime_error("could not open file for write " + filename);
  }
  saveSound<T>(f, input, samplerate, channels, format, subformat);
}

template <typename T>
void saveSound(
    std::ostream& f,
    const std::vector<T>& input,
    int64_t samplerate,
    int64_t channels,
    SoundFormat format,
    SoundSubFormat subformat) {
  SF_VIRTUAL_IO vsf = {sf_vio_wo_get_filelen,
                       sf_vio_wo_seek,
                       sf_vio_wo_read,
                       sf_vio_wo_write,
                       sf_vio_wo_tell};
  SNDFILE* file;
  SF_INFO info;

  if (formats.find(format) == formats.end()) {
    throw std::invalid_argument("saveSound: invalid format");
  }
  if (subformats.find(subformat) == subformats.end()) {
    throw std::invalid_argument("saveSound: invalid subformat");
  }

  info.channels = channels;
  info.samplerate = samplerate;
  info.format =
      formats.find(format)->second | subformats.find(subformat)->second;

  if (!(file = sf_open_virtual(&vsf, SFM_WRITE, &info, &f))) {
    throw std::runtime_error(
        "saveSound: invalid format or could not write stream");
  }

  /* Circumvent a bug in Vorbis with large buffers */
  sf_count_t remainCount = input.size() / channels;
  sf_count_t offsetCount = 0;
  const sf_count_t chunkSize = 65536;
  while (remainCount > 0) {
    sf_count_t writableCount = std::min(chunkSize, remainCount);
    sf_count_t writtenCount = 0;
    if (std::is_same<T, float>::value) {
      writtenCount = sf_writef_float(
          file,
          const_cast<float*>(reinterpret_cast<const float*>(input.data())) +
              offsetCount * channels,
          writableCount);
    } else if (std::is_same<T, double>::value) {
      writtenCount = sf_writef_double(
          file,
          const_cast<double*>(reinterpret_cast<const double*>(input.data())) +
              offsetCount * channels,
          writableCount);
    } else if (std::is_same<T, int>::value) {
      writtenCount = sf_writef_int(
          file,
          const_cast<int*>(reinterpret_cast<const int*>(input.data())) +
              offsetCount * channels,
          writableCount);
    } else if (std::is_same<T, short>::value) {
      writtenCount = sf_writef_short(
          file,
          const_cast<short*>(reinterpret_cast<const short*>(input.data())) +
              offsetCount * channels,
          writableCount);
    } else {
      throw std::logic_error("saveSound: called with unsupported T");
    }
    if (writtenCount != writableCount) {
      sf_close(file);
      throw std::runtime_error("saveSound: write error");
    }
    remainCount -= writtenCount;
    offsetCount += writtenCount;
  }
  sf_close(file);
}

template std::vector<float> loadSound(const std::string&);
template std::vector<double> loadSound(const std::string&);
template std::vector<int> loadSound(const std::string&);
template std::vector<short> loadSound(const std::string&);

template std::vector<float> loadSound<float>(std::istream&);
template std::vector<double> loadSound<double>(std::istream&);
template std::vector<int> loadSound<int>(std::istream&);
template std::vector<short> loadSound<short>(std::istream&);

template void saveSound(
    const std::string&,
    const std::vector<float>&,
    int64_t,
    int64_t,
    SoundFormat,
    SoundSubFormat);
template void saveSound(
    const std::string&,
    const std::vector<double>&,
    int64_t,
    int64_t,
    SoundFormat,
    SoundSubFormat);
template void saveSound(
    const std::string&,
    const std::vector<int>&,
    int64_t,
    int64_t,
    SoundFormat,
    SoundSubFormat);
template void saveSound(
    const std::string&,
    const std::vector<short>&,
    int64_t,
    int64_t,
    SoundFormat,
    SoundSubFormat);
}
}
}
