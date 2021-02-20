#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from .flashlight_lib_sequence_criterion import (
    CpuForceAlignmentCriterion,
    CpuFullConnectionCriterion,
    CpuViterbiPath,
    CriterionScaleMode,
)


have_torch = False
try:
    import torch

    have_torch = True
except ImportError:
    pass

if have_torch:
    from flashlight.lib.sequence.criterion_torch import (
        ASGLoss,
        FCCFunction,
        FACFunction,
        check_tensor,
        create_workspace,
        get_cuda_stream_as_bytes,
        get_data_ptr_as_bytes,
        run_backward,
        run_direction,
        run_get_workspace_size,
        run_forward,
    )
