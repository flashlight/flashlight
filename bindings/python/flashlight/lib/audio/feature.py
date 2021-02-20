#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from .flashlight_lib_audio_feature import (
    Ceplifter,
    Dct,
    Derivatives,
    Dither,
    FeatureParams,
    FrequencyScale,
    Mfcc,
    Mfsc,
    PowerSpectrum,
    PreEmphasis,
    TriFilterbank,
    Windowing,
    WindowType,
    cblas_gemm,
    frame_signal,
)


# Not sure why this is needed. Avoids this error on FB cluster:
# Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so.
try:
    import libfb.py.mkl
except ImportError:
    pass
