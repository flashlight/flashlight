"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import itertools
from pathlib import Path


def load_data(filename):
    with open(filename) as f:
        return [
            float(x) for x in itertools.chain.from_iterable(line.split() for line in f)
        ]


def test_mfcc():
    from flashlight.lib.audio import feature as fl_feat

    data_path = (
        Path(__file__)
        .absolute()
        .parent.joinpath("../../../flashlight/lib/audio/test/feature/data")
    )

    wavinput = load_data(data_path.joinpath("sa1.dat"))
    # golden features to compare
    htkfeatures = load_data(data_path.joinpath("sa1-mfcc.htk"))

    assert len(wavinput) > 0
    assert len(htkfeatures) > 0

    params = fl_feat.FeatureParams()
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
    mfcc = fl_feat.Mfcc(params)
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

    assert max(differences) < 0.4
    assert sum(differences) / len(differences) < 0.03
