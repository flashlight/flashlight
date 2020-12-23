#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import itertools as it
import os
import sys

from flashlight.lib.audio.feature import FeatureParams, Mfcc


def load_data(filename):
    path = os.path.join(data_path, filename)
    path = os.path.abspath(path)
    with open(path) as f:
        return [float(x) for x in it.chain.from_iterable(line.split() for line in f)]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} feature_test_data_path", file=sys.stderr)
        print(
            "  (usually: <flashlight_root>/lib/test/audio/feature/data)",
            file=sys.stderr,
        )
        sys.exit(1)

    data_path = sys.argv[1]

    wavinput = load_data("sa1.dat")
    # golden features to compare
    htkfeatures = load_data("sa1-mfcc.htk")

    assert len(wavinput) > 0
    assert len(htkfeatures) > 0

    params = FeatureParams()
    # define parameters of the featurization
    params.sampling_freq = 16000
    params.low_freq_filterbank = 0
    params.high_freq_filterbank = 8000
    params.num_filterbank_chans = 20
    params.num_cepstral_coeffs = 13
    params.use_energy = False
    params.zero_mean_frame = False
    params.use_power = False

    # apply MFCC featurization
    mfcc = Mfcc(params)
    features = mfcc.apply(wavinput)

    # check that obtained features are the same as golden one
    assert len(features) == len(htkfeatures)
    assert len(features) % 39 == 0
    numframes = len(features) // 39
    featurescopy = features.copy()
    for f in range(numframes):
        for i in range(1, 39):
            features[f * 39 + i - 1] = features[f * 39 + i]
        features[f * 39 + 12] = featurescopy[f * 39 + 0]
        features[f * 39 + 25] = featurescopy[f * 39 + 13]
        features[f * 39 + 38] = featurescopy[f * 39 + 26]
    differences = [abs(x[0] - x[1]) for x in zip(features, htkfeatures)]

    print(f"max_diff={max(differences)}")
    print(f"avg_diff={sum(differences)/len(differences)}")
