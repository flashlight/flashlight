# Getting Started with Automatic Speech Recognition in Flashlight

This tutorial uses the following binaries with the following capabilities:
- `fl_asr_tutorial_inference_ctc`: perform inference with an existing model with CTC loss
- `fl_asr_tutorial_finetune_ctc`: finetune an existing CTC model with additional data
- `fl_asr_align`: force align audio and transcriptions using a CTC model
- `fl_asr_voice_activity_detection_ctc`: [coming soon] detect speech and perform general audio analysis

## Inference with an Existing CTC Model

See this colab notebook for a step-by-step tutorial.

The [`fl_asr_tutorial_inference_ctc`](https://github.com/facebookresearch/flashlight/blob/master/flashlight/app/asr/tutorial/InferenceCTC.cpp) binary provides a way to perform inference with CTC-trained acoustic models. To perform inference, you'll need the following components (with their corresponding `flags`):
- An acoustic model (AM) (`am_path`)
- A token set with which the AM was trained (`tokens_path`)
- A lexicon (`lexicon_path`)
- A language model for decoding (`lm_path`)

The following parameters are also configurable when performing inference:
- The sample rate of input audio (`sample_rate`)
- The beam size when decoding (`beam_size`)
- The beam size of the token beam when decoding (`beam_size_token`)
- The beam threshold (`beam_threshold`)
- The LM weight score for decoding (`lm_weight`)
- The word score for decoding (`word_score`).

See the [complete ASR app documentation](https://github.com/jacobkahn/flashlight/blob/tutorial_docs/flashlight/app/asr/README.md) for a more detailed explanation of each of these flags. See the aforementioned colab tutorial for sensible values used in a demo.

## Finetuning with an Existing CTC Model

See this colab notebook for a step-by-step tutorial.

The [`fl_asr_tutorial_finetune_ctc`](https://github.com/jacobkahn/flashlight/blob/tutorial_docs/flashlight/app/asr/tutorial/FinetuneCTC.cpp) binary provides a means of finetuning an existing trained acoustic model on additional labeled audio. Usage of the binary is as follows:
```
fl_asr_tutorial_finetune_ctc [path to directory containing model] [...flags]
```
To finetune, you'll need the following components (with their corresponding `flags`):
- An acoustic model (AM) to finetune (the first argument to the binary invocation, e.g. `fl_asr_tutorial_finetune_ctc [path] [...flags]`)
- A token set with which the AM was trained (`tokens`)*
- A lexicon (`lexicon`)
- Validation sets to use for finetuning (`valid`)
- Train sets with data on which to finetune (`train`)

* Should be identical to that with which the original AM was trained. Will be provided with the AM in recipes/tutorials.

See the [complete ASR app documentation](https://github.com/jacobkahn/flashlight/blob/tutorial_docs/flashlight/app/asr/README.md) for a more detailed explanation of each of these flags. See the aforementioned colab tutorial for robust pre-trained models and their accompanying components that can be easily used for finetuning. The [wav2letter Robust ASR (RASR) recipe](https://github.com/facebookresearch/wav2letter/tree/master/recipes/rasr) contains robust pre-trained models and resources for finetuning.
