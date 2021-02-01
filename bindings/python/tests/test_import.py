"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import os


def test_import_lib_audio():
    from flashlight.lib.audio import feature as fl_feat


def test_import_lib_sequence():
    from flashlight.lib.sequence import criterion as fl_crit


def test_import_lib_text():
    from flashlight.lib.text import dictionary as fl_dict

    if os.getenv("USE_KENLM", "").upper() not in ["OFF", "0", "NO", "FALSE", "N"]:
        from flashlight.lib.text import decoder as fl_decoder
