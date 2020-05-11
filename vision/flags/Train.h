#pragma once

#include <gflags/gflags.h>

namespace fl {
namespace cv {
namespace flags {

// General training parameters
DECLARE_string(data_dir);
DECLARE_uint64(epochs);
DECLARE_uint64(batch_size);
// Optimizer
DECLARE_double(lr);
DECLARE_double(momentum);
DECLARE_double(wd);

// Distributed training
DECLARE_int64(world_rank);
DECLARE_int64(world_size);
DECLARE_string(rndv_filepath);

// Model saving and loading
DECLARE_string(checkpointpath);
DECLARE_int64(checkpoint);

} // end namespace flags
} // end namespace cv
} // end namespace fl
