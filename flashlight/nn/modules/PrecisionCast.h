/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <flashlight/autograd/Variable.h>
#include <flashlight/common/Serialization.h>
#include <flashlight/common/Utils.h>
#include <flashlight/nn/modules/Module.h>

namespace fl {
/**
 * Precision Cast Module. Casts the input from its original precision to the
 * target precision. Precision cast only alters the underlying array and leaves
 * other attributes of the input variable unchanged.
 */
class PrecisionCast : public Module {
  af::dtype targetType_;
  PrecisionCast() = default;
  FL_SAVE_LOAD_WITH_BASE(Module, targetType_)
 public:
  PrecisionCast(af::dtype targetType);
  std::vector<Variable> forward(const std::vector<Variable>& inputs) override;
  Variable forward(const Variable& input);
  Variable operator()(const Variable& input);
  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::PrecisionCast)