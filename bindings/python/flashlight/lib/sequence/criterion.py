#!/usr/bin/env python3

from flashlight._lib_sequence_criterion import (
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
