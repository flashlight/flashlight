 # Flashlight Applications

 Flashlight application libraries are domain-specific libraries build on top of the [flashlight core](https://github.com/flashlight/flashlight/tree/master/flashlight/fl) and [flashlight lib](https://github.com/flashlight/flashlight/tree/master/flashlight/lib). They provide lightweight, unopinionated pipelines and tools that are easily modifiable for training or inference across tasks. Below are supported applications; new applications are under active development.

 ### [Automatic Speech Recognition](https://github.com/flashlight/flashlight/tree/master/flashlight/app/asr) — `asr` (the [wav2letter](https://github.com/flashlight/wav2letter/) Project)

 The `asr` application provides tools for audio processing/augmentation, acoustic model training, beam search decoding, and preprocessing/preparing audio data for use. Full documentation for usage and binaries [can be found here](https://github.com/flashlight/flashlight/tree/master/flashlight/app/asr).

 #### Provided Artifacts:
- Binaries:
  - `fl_asr_train`
  - `fl_asr_test`
  - `fl_asr_decode`
- Tutorial/Utility Binaries
  - `fl_asr_tutorial_inference_ctc`
  - `fl_asr_tutorial_finetune_ctc`

### [Language Modeling](https://github.com/flashlight/flashlight/tree/master/flashlight/app/lm) — `lm`

The `lm` application provides tools for text preprocessing and language model training for both auto-regressive and BERT-style models. Full documentation for usage and binaries [can be found here](https://github.com/flashlight/flashlight/tree/master/flashlight/app/lm).

#### Provided Artifacts:
- Binaries:
  - `fl_lm_dictionary_builder`
  - `fl_lm_train`

### [Image Classification](https://github.com/flashlight/flashlight/tree/master/flashlight/app/imgclass) — `imgclass`

The `imgclass` application is still in early, active development. It currently provides dataset abstractions for ImageNet and example training pipelines for `Resnet34` and `ViT` (vision transformer) which can be easily extended to more complex setups.

#### Provided Artifacts
- Binaries:
  - `fl_img_imagenet_resnet34`
  - `fl_img_imagenet_vit`
  - `fl_img_eval`

### [Object Detection](https://github.com/flashlight/flashlight/tree/master/flashlight/app/objdet) — `objdet`

The `objdet` application provides tools for objet detection. It is still under active development, but currently provides dataset abstractions for COCO and an example training pipeline for `DETR`.

#### Provided Artifacts
- Binaries:
  - `fl_img_coco_detr`
