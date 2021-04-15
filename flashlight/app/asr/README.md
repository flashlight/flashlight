# Automatic Speech Recognition (ASR)

Flashlight's ASR application (formerly the [wav2letter](https://github.com/flashlight/wav2letter/) project) provides training and inference capabilities for end-to-end speech recognition systems. Outside of original research conducted with Flashlight and wav2letter, the codebase contains up-to-date implementations of recent architectures and developments in the speech domain.

**To get started using the ASR library with existing/pre-trained models, see [tutorials](https://github.com/flashlight/flashlight/tree/master/flashlight/app/asr/tutorial).**

### Table of Contents

- [Structure](#structure)
  * [Notation](#notation)
- [Training acoustic models (AMs)](#training-acoustic-models--ams-)
  * [Data Preparation](#data-preparation)
    + [Audio and transcriptions data](#audio-and-transcriptions-data)
    + [Token dictionary](#token-dictionary)
    + [Lexicon File](#lexicon-file)
  * [Writing architecture files](#writing-architecture-files)
  * [How to train acoustic model](#how-to-train-acoustic-model)
    + [Flags](#flags)
  * [Batching strategy](#batching-strategy)
  * [Distributed training](#distributed-training)
- [Beam-search decoder](#beam-search-decoder)
  * [Greedy Decoder](#greedy-decoder)
  * [Beam-search Decoders](#beam-search-decoders)
    + [1. Beam-search Decoder Types](#1-beam-search-decoder-types)
      - [1.1 Lexicon-based beam-search decoder (`uselexicon=true`)](#11-lexicon-based-beam-search-decoder---uselexicon-true--)
      - [1.2 Lexicon-free beam-search decoder (`uselexicon=false`)](#12-lexicon-free-beam-search-decoder---uselexicon-false--)
    + [2. Beam-search Optimization](#2-beam-search-optimization)
    + [3. Language Model Incorporation](#3-language-model-incorporation)
      - [3.1 Level of language model tokens](#31-level-of-language-model-tokens)
      - [3.2 Types of language models](#32-types-of-language-models)
    + [4. Distributed Decoding](#4-distributed-decoding)
    + [5. Online beam-search decoding](#5-online-beam-search-decoding)
  * [Full list of flags related to the beam-search decoding](#full-list-of-flags-related-to-the-beam-search-decoding)
    + [Beam-search decoder options to be specified (with examples)](#beam-search-decoder-options-to-be-specified--with-examples-)
    + [Common flags](#common-flags)
    + [Flags related to the beam-search algorithm](#flags-related-to-the-beam-search-algorithm)
    + [Parameters to optimize for beam-search decoder](#parameters-to-optimize-for-beam-search-decoder)
  * [Template to run beam-search decoder](#template-to-run-beam-search-decoder)
    + [Configuration file support](#configuration-file-support)

## Structure
Binaries:
- `Train.cpp` is compiled into `build/bin/fl_asr_train`, used to train an acoustic model
- `Test.cpp` is compiled into `build/bin/fl_asr_test`, used to transcribe audio with the argmax path + serialize emissions matrix for input data on the disk.
- `Decode.cpp` is compiled into `build/bin/fl_asr_decode`, used to transcribe audio with integration of a language model via the beam-search decoder.

Code:
- `common` defines constants and cmd flags
- `criterion` contains criterions (loss functions): ASG, CTC and seq2seq with attention
- `data` contains speech datasets and other dataset utilities
- `decoder` contains utilities for the beam-search decoder, like transformation between indices and tokens
- `runtime` contains helper functions, like logging, dataset creation (Note: here FLAGS are used)

### Notation

* AM - acoustic model
* LM - language model
* WER - Word Error Rate
* LER - Letter Error Rate

## Training acoustic models (AMs)

### Data Preparation
For training ASR model, we expect the following inputs
- Audio and their transcriptions data
- Token dictionary (acoustic model predicts distribution over tokens for input audio frames)
- Lexicon (mapping of words into tokens sequence)

#### Audio and transcriptions data
ASR binaries expect audio and transcription data to be prepared in a specific format.
To specify audio data with their transcriptions we are using a similar approach as for vision domain: each dataset (test/valid/train) is defined by a text file (list file) where each line defines one sample. A sample is specified using 4 columns separated by space (or tabs).
 - `sample_id` - unique id for the sample
 - `input_handle` - input audio **absolute** file path
 - `input_size` - a real number used for sorting the dataset for efficient packing into batches (typically audio duration in milliseconds)
 - `transcription` - target word transcription for this sample

Example of list file `my.lst`:
```
# sample_id input_handle input_size  transcription
train001 /tmp/000000000.flac 100.03  this is sparta
train002 /tmp/000000001.flac 360.57  coca cola
train003 /tmp/000000002.flac 123.53  hello world
train004 /tmp/000000003.flac 999.99  quick, brown, fox, jumped!
...
```

To specify training, validation and test data use flags `train`, `valid`, `test` correspondingly. Flag `datadir` is used as prefix for the list file paths, for example `--datadir=/tmp --train=my.lst --valid=eval/myeval.lst` will read list files `/tmp/my.lst` as train and `/tmp/eval/mueval.lst` as validation sets.

We use [sndfile](https://github.com/erikd/libsndfile/) for loading the audio files. It supports many different formats including `.wav`, `.flac` etc.
For sample rate, 16KHz is the default option but you can specify a different one using `samplerate` flag. Note that, we require all the train/valid/test data to have the same `samplerate` for now.

We also support empty transcription which can be used for inference for example with test and decode binaries.

#### Token dictionary

A token dictionary file consists of a list of all subword units (phonemes / graphemes / word pieces / words) used to train acoustic models. Acoustic model will return distribution over tokens set for each time frame. If we are using graphemes, a typical token dictionary file would look like this

```
# tokens.txt
|
'
a
b
c
...
... (and so on)
z
```

The symbol "|" is used to denote space between words. Note that it is possible to add additional symbols like `N` for noise, `L` for laughter and more depending on the dataset. Tokens file defines the mapping between a token and its index in the emission matrix (in the output). If two tokens are on the same line in the tokens file, they are mapped to the same index for training/decoding.

Tokens file is specified via `tokens` flag which specifies full path to the tokens file, for example `--tokens=/tmp/tokens.txt will read tokens set from `/tmp/tokens/tokens.txt` file.

#### Lexicon File

A lexicon file (configured via `lexicon` flag) consists of mapping from words to their sequence of tokens representation.
Each line will have a word followed by its space-split tokens. Space or tab separate a word from its sequence of tokens. Here is an example of grapheme based lexicon:
```
# lexicon.txt
a a |
able a b l e |
about a b o u t |
above a b o v e |
...
hello-kitty h e l l o | k i t t y |
...
... (and so on)
```
Here we also use `|` in the mapping to have correct insertion of word boundaries. Then you also need to provide flag `wordseparator` to give information about what token is used as word boundary to correctly map back when WER and LER are computed (for our example it should be `--wordseparator=|`).
For the following sample
```
train001 /tmp/000000000.flac 100.03  this is sparta
```
we will have tokens sequence using the lexicon file `t h i s | i s | s p a r t a |`. One can use also the `surround` flag to add additional tokens at the beginning and end of each token sequence for the sample. So if `surround=N` then the target tokens sequence will be `N t h i s | i s | s p a r t a | N`. In case `surround` is the same as `wordseparator` there will be no duplication of `wordseparator` in the beginning and at the end.

If word separator token is not a separate token, but could be also part of another token (for example If one uses [sentencepiece](https://github.com/google/sentencepiece) tool to prepare the word-pieces tokens) then you need to specify `--usewordpiece=true` which will check additionally `wordseparator` token inside tokens themselves when computes WER and LER.

In case you have mapping to tokens which are not presented in the tokens file they will be skipped for target construction. Out-of-vocabulary words, which are not present in the lexicon, will be mapped into graphemes sequence with adding the `wordseparator` (depending on `usewordpiece` flag will be added at the beginning or at the end of the sequence) and then these graphemes will be checked on the presence in the tokens set (not presented graphemes will be skipped).

### Writing architecture files

For now we provide a simple way to create a `fl::Sequential` module for the acoustic model from text files. These are specified using the flags `arch` (file path with the architecture).

Example of architecture file:
```
# Comments like this are ignored
# the output tensor will have the shape (Time, 1, NFEAT, Batch)
# View
V -1 1 NFEAT 0
# Convolution layer
C2 NFEAT 300 48 1 2 1 -1 -1
# ReLU activation
R
C2 300 300 32 1 1 1
R
# Reorder
RO 2 0 3 1
# the output should be with the shape (NLABEL, Time, Batch, 1)
# Linear layer
L 300 NLABEL
```

While parsing, we ignore lines starting with `#` as comments. We also replace the following tokens `NFEAT` = input feature size (e.g. number of frequency bins), `NLABEL` = output size (e.g. number of grapheme tokens)

The first token in each line represents a specific `flashlight` module followed by the specification of its parameters.

<details><summary>Here, we describe how to specify different `flashlight` modules in the architecture files.</summary>

**fl::Conv2D**
```
C2 [inputChannels] [outputChannels] [xFilterSz] [yFilterSz] [xStride] [yStride] [xPadding <OPTIONAL>] [yPadding <OPTIONAL>] [xDilation <OPTIONAL>] [yDilation <OPTIONAL>]
```
Input is expected to be `[Time, Width=1, inputChannels, Batch]`, and the output `[Time, Width=1, outputChannels, Batch]`.

*Use padding `= -1` for `fl::PaddingMode::SAME`*.

**fl::Linear**
```
L [inputChannels] [outputChannels]
```
Input is expected to be `[inputChannels, *, * , *]`, and the output `[outputChannels, *, * , *]`.

**fl::BatchNorm**
```
BN [totalFeatSize] [firstDim] [secondDim <OPTIONAL>] [thirdDim <OPTIONAL>]
```
Dimensions which are not presented in the list will be reduced for statistics computation.

**fl::LayerNorm**
```
LN [firstDim] [secondDim <OPTIONAL>] [thirdDim <OPTIONAL>]
```
Dimensions along which normalization is computed (these axes will be reduced for statistics computation).

**fl::WeightNorm**
```
WN [normDim] [Layer]
```

**fl::Dropout**
```
DO [dropProb]
```

**fl::Pool2D**
   1. Average
      ```
      A [xFilterSz] [yFilterSz] [xStride] [yStride] [xPadding] [yPadding]
      ```
   1. Max
      ```
      M [xFilterSz] [yFilterSz] [xStride] [yStride] [xPadding] [yPadding]
      ```

*Use padding `= -1` for `fl::PaddingMode::SAME`*.

**fl::View**
```
V [firstDim] [secondDim] [thirdDim] [fourthDim]
```
*Use `-1` to infer dimension, only one param can be a `-1`. Use `0` to use the corresponding input dimension*.

**fl::Reorder**
```
RO [firstDim] [secondDim] [thirdDim] [fourthDim]
```

**fl::ELU**
```
ELU
```

**fl::ReLU**
```
R
```

**fl::PReLU**
```
PR [numElements <OPTIONAL>] [initValue <OPTIONAL>]
```

**fl::Log**
```
LG
```

**fl::HardTanh**
```
HT
```

**fl::Tanh**
```
T
```

**fl::GatedLinearUnit**
```
GLU [sliceDim]
```

**fl::LogSoftmax**
```
LSM [normDim]
```

**fl::RNN**
   1. RNN
      ```
      RNN [inputSize] [outputSize] [numLayers] [isBidirectional] [dropProb]
      ```
   1. GRU
      ```
      GRU [inputSize] [outputSize] [numLayers] [isBidirectional] [dropProb]
      ```
   1. LSTM
      ```
      LSTM [inputSize] [outputSize] [numLayers] [isBidirectional] [dropProb]
      ```

**fl::Embedding**
```
E [embeddingSize] [nTokens]
```

**fl::AdaptiveEmbedding**
```
ADAPTIVEE [embeddingSize] [cutoffs]
```
Cuttofs must be strictly ascending. For example `10,50,100` means 3 groups of embeddings for 10, 40 and 50 tokens. Total tokens is 100.

**fl::SinusoidalPositionEmbedding** (sinusoidal absolute embedding)
```
SINPOSEMB [embeddingSize] [inputScale <OPTIONAL, DEFAULT=1.>]
```

**fl::PositionEmbedding** (learnable absolute positional embedding)
```
POSEMB [embeddingSize] [contextSize] [dropout <OPTIONAL, DEFAULT=0.>]
```

**fl::AsymmetricConv1D**
```
AC [inputChannels] [outputChannels] [xFilterSz] [xStride] [xPadding <OPTIONAL>] [xFuturePart <OPTIONAL>] [xDilation <OPTIONAL>]
```
Input is expected to be `[Time, Width=1, inputChannels, Batch]`, and the output `[Time, Width=1, outputChannels, Batch]`.

**fl::Residual**
```
RES [numLayers (N)] [numResSkipConnections (K)] [numBlocks <OPTIONAL>]
[Layer1]
[Layer2]
[ResSkipConnection1]
[Layer3]
[ResSkipConnection2]
[Layer4]
...
[LayerN]
...
[ResSkipConnectionK]
```

Residual skip connections between layers can only be added if these layers have already been added. There two ways to define residual skip connection:
- standard
```
SKIP [fromLayerInd] [toLayerInd] [scale <OPTIONAL, DEFAULT=1>]
```
- with a sequence of projection layers, when, for the residual skip connection, the number of channels in the output of `fromLayer` differs from the number of channels expected in the input of `toLayer` (or some transformation is needed to be applied):
```
SKIPL [fromLayerInd] [toLayerInd] [nLayersInProjection (M)] [scale <OPTIONAL, DEFAULT=1>]
[Layer1]
[Layer2]
...
[LayerM]
```
where `scale` is the value by which the final output is multiplied (`(x + f(x)) * scale`). `scale` must be the same for all residual skip connections that share the same `toLayer`.
*(Use fromLayerInd `= 0` for a skip connection from input, toLayerInd `= N+1` for a residual skip connection to output, and fromLayerInd/toLayerInd `= K` for a residual skip connection from/to LayerK.)*

**fl::TDS**
```
TDS [inputChannels] [kernelWidth] [inputWidth] [dropoutProb <OPTIONAL, DEFAULT=0>] [innerLinearDim <OPTIONAL, DEFAULT=0>] [rightPadding <OPTIONAL, DEFAULT=-1>] [lNormIncludeTime <OPTIONAL, DEFAULT=True>]
```
Description of these params can be found [here](https://git.io/Jvfhk)

**fl::PADDING**
```
PD [value] [kernelWidth] [dim0PadBefore] [dim0PadAfter] [dim1PadBefore <OPTIONAL, DEFAULT=0>] [dim1PadAfter <OPTIONAL, DEFAULT=0>] [dim2PadBefore <OPTIONAL, DEFAULT=0>] [dim2PadAfter <OPTIONAL, DEFAULT=0>] [dim3PadBefore <OPTIONAL, DEFAULT=0>] [dim3PadAfter <OPTIONAL, DEFAULT=0>]
```

**fl::Transformer** (for encoder/decoder blocks)
```
TR [embeddingDim] [mlpDim] [nHeads] [maxContext] [dropout] [layerDropout <OPTIONAL, DEFAULT=0>] [usePreNormLayer <OPTIONAL, DEFAULT=False>] [useFutureMask <OPTIONAL, DEFAULT=False>]
```
maxContext is often max time dimension.

**fl::Conformer** (encoder block)
```
CFR [embeddingDim] [mlpDim] [nHeads] [maxContext] [ConvKernel] [dropout] [layerDropout <OPTIONAL, DEFAULT=0>]
maxContext is often max time dimension.
```
**fl::PrecisionCast** (casting layer)
```
PC [afTypeString]
```
</details>

### How to train acoustic model
Training is provided with Train.cpp which is compiled into binary `build/bin/fl_asr_train`. It supports three modes
- `train` : Train a model from scratch on the given training data.
- `continue` : Continue training a saved model. This can be used for example to
  fine-tune with a smaller learning rate. The `continue` option makes a best
  effort to resume training from the most recent checkpoint of a given model as
  if there were no interruptions. Both network and optimizer states will be reloaded.
- `fork` : Create and train a new model from a saved model. This can be used
  for example to adapt a saved model to a new dataset. Only network state will be reused and a new optimizer will be recreated

At its simplest, training a model can be invoked with
```
<build/bin/fl_asr_train> train --flagsfile=<path_to_flags>
```
The flags to the train binary can be passed in a flags file or as flags on the command line. Example of flags file:
```
--dadadir=/tmp
--train=train.lst
...
```
Example of command line:
```
<build/bin/fl_asr_train> [train|continue|fork] \
--datadir=<path/to/data/> \
--tokens=<path/to/tokens/file> \
--arch=<path/to/architecture/file> \
--rundir=<path/to/save/models/> \
--train=<train/datasets/ds1.lst,train/datasets/ds2.lst> \
--valid=<validation/datasets/ds1.lst,validation/datasets/ds2.lst> \
--lexicon=<path/to/lexicon.txt> \
--lr=0.0001 \
--lrcrit=0.0001
```
In case of `continue` or `fork` modes the path to the model snapshot also should be specified with `am` flag.

The log of training looks like
```
epoch:        6 | nupdates:         1000 | lr: 0.000469 | lrcriterion: 0.000000 | runtime: 00:10:23 | bch(ms): 623.52 | smp(ms): 2.27 | fwd(ms): 213.36 | crit-fwd(ms): 15.79 | bwd(ms): 365.71 | optim(ms): 41.55 | loss:   34.22960 | train-LER: 77.16 | train-WER: 100.79 | dev-clean-loss:   19.73666 | dev-clean-LER: 57.10 | dev-clean-WER: 103.32 | dev-other-loss:   20.93873 | dev-other-LER: 62.13 | dev-other-WER: 104.02 | test-clean-loss:   19.91433 | test-clean-LER: 57.36 | test-clean-WER: 105.57 | test-other-loss:   21.04050 | test-other-LER: 62.48 | test-other-WER: 104.15 | avg-isz: 1263 | avg-tsz: 216 | max-tsz: 323 | hrs:  561.64 | thrpt(sec/sec): 3242.74
```
where we report epochs, number of updates, learning rates, timing, loss and WER/LER for train and validation data.

#### Flags
We give a short description of some of the more important flags here. A complete list of the flag definitions and short descriptions of their meaning can be found [here](https://github.com/flashlight/flashlight/blob/master/flashlight/app/asr/common/Defines.cpp).

The `datadir` flag is the base path to where all the `train` and `valid` dataset list files live. Every `train` path will be prefixed by `datadir`. Multiple datasets can be passed to `train` and `valid` as a comma-separated list.

Similarly, the `arch` and `tokens` are the paths to architecture file and tokens file.  `lexicon` flag is used to specify the lexicon which specifies the token sequence for a given word.

The `rundir` flag is the checkpoint directory that will be created to save the model and training logs.

Most of the training hyperparameter flags have default values. Many of these you will not need to change. Some of the more important ones include:
- `lr` : The learning rate for the model parameters.
- `lrcrit` : The learning rate for the criterion parameters.
- `criterion` : Which criterion (e.g. loss function) to use. Options include `ctc`,
  `asg` or `seq2seq`.
- `batchsize` : The size of the minibatch to use per GPU.
- `maxgradnorm` : Clip the norm of gradient of the model and criterion parameters
  to this value. NB the norm is computed and clipped on the aggregated model
  and criterion parameters.

### Batching strategy
Before batching the input data we sort them by descending order of audio sizes (this is done for efficient packing with a small amount of padding). Each audio in the batch is padded with zero to the max sample size after featurization (computation of MFCC, etc.). After we packed all data into batches we never change these batches themselves. Before a new epoch we only shuffle batches indices (not the data between batches).

### Distributed training
We support distributed training on multiple GPUs out of the box. To run on multiple GPUs set pass the flag `--enable_distributed=true` and run with MPI:
```
mpirun <build/bin/fl_asr_train> -n 8 <> [train|continue|fork] \
--enable_distributed=true \
<... other flags ..>
```

The above command will run data parallel training with 8 processes (e.g. on 8 GPUs).

## Beam-search decoder
After acoustic model (AM) is trained, one can get the transcription of a audio by running either the **greedy decoder** (the greedy best path using only acoustic model predictions, in the code Viterbi name is used) or the **beam-search decoding** with language model (LM) incorporated.

### Greedy Decoder
It is implemented in the `Test.cpp` which is compiled into binary `build/bin/fl_asr_test`.
To get the greedy decoder, one should use the **Test binary** in the following way

```sh
build/bin/fl_asr_test \
  --am=path/to/train/am.bin \
  --maxload=10 \
  --test=path/to/test/list/file
```

For this particular example, greedy decoder will be computed on 10 random samples (`--maxload=10`) from the test list file and WER and LER will be printed on the screen. To run on all samples, set `--maxload=-1`.

While running the **Test binary**, the AM is loaded and all the saved flags will be used if you don’t specify them in the command line. For example, tokens and lexicon paths. So, in case you want to overwrite them, you should directly specify them:

```sh
build/bin/fl_asr_test \
  --am=path/to/train/am.bin \
  --maxload=10 \
  --test=path/to/test/list/file \
  --tokens=path/to/tokens/file \
  --lexicon=path/to/the/lexicon/file
```

Lexicon file is used again to map words in the target transcription to the sequence of tokens and also map sequences of predicted tokens into words to compute WER. Out-of-vocabulary (OOV) words not presented in the lexicon will be always considered as errors for WER computation, that is why use `--uselexicon=false` flag which will disable usage of lexicon and will use `wordseparator` and `usewordpiece` flags to map sequence of tokens into words and take into account recognition of OOV correctly into WER computation.

The **Test binary** can be used also to generate an **Emission Set** including the emission matrix as well as other target-related information for each sample. All flags are also stored in the **Emission Set**. Specifically, the emission matrix of the CTC/ASG model is the posterior, while for seq2seq models, it is an encoded audio with a series of embeddings. The **Emission Set** can be fed into the **Decode binary** directly to generate transcripts without running AM forwarding again. To set the directory where to store **Emission Set** use the flag  `--emission_dir=path/to/emission/dir` (default value is `''`) and the `--test` will be used as a file name.

Summarization on flags to run **Test binary**:

|Flags |Flag Type |Default Value |Flag Example Value |Reused from the AM training/ Emission Set |Description |
|:---: |:---: |:---: |:---: |:---: |:---: |
|`am` |string |`''`  |`--am path/to/am/file` |N |Full path to the acoustic model binary file |
|`emission_dir` |string |`''`  |`--emission_dir path/to/emission/dir` |N |Path to the directory where emission set will be stored to prevent running the AM forward pass during beam-search decoding. |
|`datadir` |string |`''`  |`--datadir path/to/the/list/file/dir` |Y |This prefix is used to define the full path to the test list. Set it to `''` in case you specify the full path in the `--test`. |
|`test` |string |`''`  |`--test path/to/the/test/list/file` |Y |Path to the test list file (where `id path duration transcription` are stored, transcription can be empty).  `--datadir` parameter is used as prefix for this path (concatenation of paths is done) |
|`maxload` |int |-1 |`--maxload 300` |N |Number of random sample to process (value -1, means all samples) |
|`show` |bool |`false` |`--show` |N |To print word transcriptions (target and predicted) for each sample into stdout |
|`showletters` |bool |`false` |`--showletters` |N |To print token transcriptions (target and predicted) for each sample into stdout |
|`sclite` |string |`''`  |`--sclite path/to/file` |N |Specifies the path to save the logs, including the *stdout* log and the hypotheses and references in *sclite* format ([trn](http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/infmts.htm#trn_fmt_name_0)) |
|`uselexicon` |bool |`true` |`--uselexicon false` |N |Use or not lexicon for mapping between words and tokens sequence. In case of `false` flags `wordseparator` and `usewordpiece` are used to do mapping. |


### Beam-search Decoders
It is implemented in the `Decode.cpp` which is compiled into binary `build/bin/fl_asr_decode`.

#### 1. Beam-search Decoder Types

We support lexicon-base beam-search decoder and lexicon-free beam-search decoder for acoustic models trained with CTC, ASG and Seq2Seq criterion.

##### 1.1 Lexicon-based beam-search decoder (`uselexicon=true`)
For a lexicon-based decoder we restrict our search by the lexicon provided by a user. In other words, generated transcriptions only contain words from the lexicon.

The lexicon is a mapping from words to their tokens sequence, spellings. The tokens set should be identical to the one used in AM training. For example, if we train AM with letters as a token set `{a-z}` then the word “world” should have the spelling “w o r l d”.

To optimize the decoding performance, spellings of the words are stored in a `Trie`. Each node in the `Trie` corresponds to a token. Some nodes, usually the leaf nodes, represent valid words containing the spelling of tokens on the path from `Trie` root to it. In case we have “hello”, “world“, ”pineapple“, ”pine“ in the lexicon and letters are our tokens set we will have as a trie:

```
root → h → e → l → l → o ([hello])
root → w → o → r → l → d ([world])
root → p → i → n → e ([pine]) → a → p → p → l → e → ([pineapple])
```

**Note** We forced to have up to 6 words with the same spelling, others will be ignored in the inference. So if you have more than 6 words in the lexicon, you need to [update this constant](https://github.com/flashlight/flashlight/blob/master/flashlight/lib/text/decoder/Trie.h#L17).

##### 1.2 Lexicon-free beam-search decoder (`uselexicon=false`)
The lexicon-free beam-search decoder considers any possible token as candidates and there is no notion of words during decoding. In this case, a word separator should be set by `wordseparator` and included into tokens set for AM training. The word separator is treated and predicted as all the other normal tokens. After obtaining the transcription in tokens, word separator is used to split the sequence into words. Usually, when we use word-pieces as target units, the word separator can be part of the token. To correctly handle this case, one should set `--usewordpiece=true`.


#### 2. Beam-search Optimization

At each decoding step, we preserve only top `beamsize` hypotheses in the beam according to their accumulated scores. Apart from the beam size, `beamthreshold` is used to limit the score range of the hypothesis in the beam, i.e. hypothesis, whose score gaps from the best are larger than this threshold, are also removed from the beam. In addition, we can also restrict the number of tokens to propose for each hypothesis. The tokens beam size `beamsizetoken` limits the search space to only the top tokens according to AM scores. This is extremely useful for lexicon free decoding, since there is no lexicon constraint.

#### 3. Language Model Incorporation
In the beam-search decoder, a language model trained with external data can be included and its scores (log-probability) will be accumulated together with AM scores.

##### 3.1 Level of language model tokens
LM can be operated on either words or tokens (which is the same as the ones used to train AM). In other words, the LM can be queried each time when a new word or token is proposed. One can set this with `decodertype`. Note that word-based LM can be used only with lexicon-based beam-search decoder, i.e. if `--decodertype=wrd` then `uselexicon` flag is ignored.

If LM is word-based, the LM score is applied only when a completed word is proposed. In order to maintain the score scale of all the hypotheses in the beam and properly rank the partial words, we approximate the LM score of partial words by their `highest` possible unigram score. This can be easily computed by recursively smear upward the trie with the real unigram scores on the nodes with valid words. Three types of smearing are supported:  `logadd` (a.k.a `logadd(a, b)=log(exp(a) + exp(b))`, `max` (pick the maximum score among children nodes scores and current node score) or `none` (no smearing). It can be set by `smearing.`

##### 3.2 Types of language models
Currently we are supporting decoding with the following language models: ZeroLM, KenLM and ConvLM. To specify LM type use `--lmtype=[kenlm, convlm]`. To use ZeroLM set the `--lm=''`.

**ZeroLM** is a fake LM which always returns 0 as score. It served as a proxy to conduct beam-search on only AM scores without breaking API.

**KenLM** language model can be trained standalone with [KenLM library](https://kheafield.com/code/kenlm/). The text data should be prepared accordingly to the acoustic model data. For example, in case of word-level LM if your AM token set doesn’t contain punctuation, then remove all punctuation from the data. In case of token-level LM training one should split words into tokens set sequence and only then train LM on such data in a way that LM predicts probability for a token (not for a word). Both of the `.arpa` and the binarized `.bin` LM can be used. However it is recommended to convert arpa files to the [binary format](https://github.com/kpu/kenlm#querying) for faster loading.

**ConvLM** models are convolutional neural networks. They are currently trained in the [fairseq](https://github.com/pytorch/fairseq) and then converted into flashlight-serializable models ([example](https://github.com/flashlight/wav2letter/blob/master/recipes/lexicon_free/librispeech/convert_convlm.sh) how we are doing this) to be able to load. `lm_vocab` should be specified as it is a dictionary to map tokens into indices in the ConvLM training. Note that this token set is usually different from the one used in AM training. Each line of this file is a single token (char, word, word-piece, etc.) and the token index is exactly its line number.

To efficiently decode with ConvLM, which is pretty expensive on running the forward pass, we design a dynamic cache to hold the probabilities over all the tokens given the candidates generated from the previous frame. This way, when we want to propose new candidates, we can easily check the cache for its pre-computed LM score. In other words, we only need to run the ConvLM forward pass in batches at the end of decoding each frame, when all the possible new candidates are gathered. Thus, the batching and caching can greatly reduce the number of the forward passes we need to run in total.

Usually, the cache has size `beam size` x `number of classes in ConvLM` in main memory. If we cannot feed `beam size` samples to ConvLM in a single batch, `lm_memory` is used to limit the size of the input batch. `lm_memory` is a integer
which requires `input batch size` x `LM context size` < `lm_memory`. For example, if the context size or receptive field of a ConvLM is 50, then no matter what the beam size or the number of new candidates is, we can only feed 100 samples in a single batch if `lm_memory` is set to `5000`.

|Flags |ZeroLM |KenLM |ConvLM |
|:---: |:---: |:---: |:---: |
|`lm` |`''` |`path/to/lm/model` |`path/to/lm/model` |
|`lmtype` |X |`kenlm` |`convlm` |
|`lm_vocab` |X |X |*V* |
|`lm_memory` |X |X |*V* |
|`decodertype` |X |*V* |*V* |

#### 4. Distributed Decoding

We support decoding a dataset using several threads by setting `nthread_decoder`. The samples in the dataset are dispatched equally to each thread. In case of decoding CTC/ASG models with KenLM language model, `nthread_decoder` is simply the number of CPU threads to run beam-search decoding. If one wants to decode Seq2Seq models or with ConvLM, we need to use `flashlight` to run forward pass in each thread. Since forwarding is not thread-safe, each thread needs to acquire resources for its own and a copy of the acoustic model (the seq2seq criterion) and LM will be stored on the device it requested. Specifically, if `flashlight` is built with CUDA backend, 1 GPU is required per thread and `nthread_decoder` should be no larger than the number of visible GPUs.

We are supporting not consumer-producer scheme for parallel computations. `nthread_decoder_am_forward` defines the number of threads for AM forward pass: all threads place forward results into the queue to process by beam-search decoder with maximum size of the queue `emission_queue_size`. In case of running forward pass on GPUs `nthread_decoder_am_forward` defines the number of GPUs to use for parallel forward pass. `nthread_decoder` threads are reading from the queue and perform beam-search decoding.


#### 5. Online beam-search decoding

Decoders, except Seq2Seq decoder, are now supporting online decoding. It consumes small chunks of emissions of audio as input. At the time we want to have a look at the transcript so far, we may get the best transcript and prune the hypothesis space and keep decoding further.


### Full list of flags related to the beam-search decoding

#### Beam-search decoder options to be specified (with examples)

|Flags |CTC / ASG criterion (Lexicon-based) |Seq2Seq criterion (Lexicon-based) |CTC / ASG criterion (Lexicon-free) |Seq2Seq criterion (Lexicon-free) |
|:---: |:---: |:---:  |:---: |:---: |
|`criterion` |`ctc` / `asg` |`seq2seq` |`ctc` / `asg` | `seq2seq` |
|`lmweight` |*V* |*V* |*V* |*V* |
|`beamsize ` |*V* |*V* |*V* |*V* |
|`beamsizetoken` |*V* |*V* |*V* |*V* |
|`beamthreshold` |*V* |*V* |*V* |*V* |
|`uselexicon` |`true` | `true` |`false` |`false` |
|`lexicon` |`path/to/the/lexicon/file` |`path/to/the/lexicon/file` |`''` | `''` |
|`smearing` |`none` / `max` / `logadd` |`none` / `max` / `logadd` |X |X |
|`wordscore` |*V* |*V* |X |X |
|`wordseparator` |X |X |*V* |*V* |
|`unkscore` |*V* |X |*V* |X |
|`silscore` |*V* |X |*V* |X |
|`eosscore` |X |*V* |X |*V* |
|`attentionthreshold` |X |*V* |X |*V* |
|`smoothingtemperature` |X |*V* |X |*V* |

#### Common flags

|Flags |Flag Type |Default Value |Flag Example Value |Reused from the AM training/ Emission Set |Description |
|:---: |:---: |:---: |:---: |:---: |:---: |
|`am` |string |`''`  |`--am path/to/am/file` |N |Full path to the acoustic model binary file. Ignored if `emission_dir` is specified. |
|`emission_dir` |string |`''`  |`--emission_dir path/to/emission/dir` |N |Path to the directory with stored emission data from the **Test binary** to prevent running the AM forward pass during beam-search decoding |
|`datadir` |string |`''`  |`--datadir path/to/the/list/file/dir` |Y |This prefix is used to define the full path to the test list. Set it to `''` in case you specify the full path in the `--test`. |
|`test` |string |`''`  |`--test path/to/the/test/list/file` |Y |Path to the test list file (where `id path duration transcription` are stored, transcription can be empty).  `--datadir` parameter is used as prefix for this path (concatenation of paths is done) |
|`maxload` |int |-1 |`--maxload 300` |N |Number of random sample to process (value -1, means all samples) |
|`show` |bool |`false` |`--show` |N |To print word transcriptions (target and predicted) for each sample into stdout |
|`showletters` |bool |`false` |`--showletters` |N |To print token transcriptions (target and predicted) for each sample into stdout |
|`nthread_decoder` |int |1 |`--nthread_decoder 4` |N |Number of threads to run beam-search decoding (details in **Distributed running** section) |
|`nthread_decoder_am_forward` |int |1 |`--nthread_decoder_am_forward 2` |N |Number of threads to run AM forward pass (details in **Distributed running** section) |
|`emission_queue_size` |int |3000 |`--emission_queue_size 1000` |N |Maximum size of the emission queue (details in **Distributed running** section) |
|`sclite` |string |`''`  |`--sclite path/to/file` |N |Specifies the path to save the logs, including the *stdout* log and the hypotheses and references in *sclite* format ([trn](http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/infmts.htm#trn_fmt_name_0)) |

#### Flags related to the beam-search algorithm

|Flags |Flag Type |Default Value |Flag Example Value |Reused from the AM training/ Emission Set |Description |
|:---: |:---: |:---: |:---: |:---: |:---: |
|``uselexicon`` |bool |`true` |`--uselexicon` |N |True to set lexicon-based beam-search decoding, false - to set lexicon-free |
|`lexicon` |string |`''` |`--lexicon path/to/the/lexicon/file` |Y |Path to the lexicon file where mapping of words into tokens is given (is used in case of lexicon-based beam-search decoding) |
|`lm` |string |`''`  |`--lm path/to/the/lm/file` |N |Full path to the language model binary file (use `''` to use zero LM) |
|`lm_vocab` |string |`''`  |`--lm_vocab path/to/lm/vocab/file` |N |Path to vocabulary file defines the mapping between indices and neural-based LM tokens |
|`lm_memory` |double |5000 |`--lm_memory 3000` |N |Total memory to define the batch size used to run forward pass for neural-based LM model |
|`lmtype` |string: `kenlm` / `convlm` |`kenlm` |`--lmtype kenlm` |N |Language model type |
|`decodertype` |string: `wrd` / `tkn` |`wrd` |`--decodertype tkn` |N |Language model token type: `wrd` for word-level LM, `tkn` - for token-level LM (tokens should be the same as an acoustic model tokens set). If `wrd` value is set then `uselexicon` flag is ignored and lexicon-based beam search decoding is used. |
|`wordseparator` |string | `\|` |`--wordseparator _` |Y |Token to be used as a separator of words (is used to get word transcription from the token transcription for the lexicon-free beam-search decoder) |
|`usewordpiece` |bool |`false` |`--usewordpiece false` |Y |Defines if acoustic model is training with tokens where word separator is not a separate token, default false (for example with word-pieces `hello world` -> `*he llo _world*`* *where* * corresponds to word separation).  |
|`smoothingtemperature` |double |1 |`--smoothingtemperature 1.2` |Y |Smoothen the posterior distribution of acoustic model (for Seq2Seq criterion only) |
|`attentionthreshold` |int |`-infinity` |`--attentionthreshold 30` |Y |Limit of the distance between the peak attention locations on the encoded audio for 2 consecutive tokens (for Seq2Seq criterion only) |

#### Parameters to optimize for beam-search decoder

|Flags |Flag Type |Default Value |Flag Example Value |Description |
|:---: |:---: |:---: |:---: |:---: |
|`beamsize ` |int |2500 |`--beamsize 100` |The number of top hypothesis to preserve at each decoding step |
|`beamsizetoken` |int |250000 |`--beamsizetoken 10` |The number of top by acoustic model scores tokens set to be considered at each decoding step |
|`beamthreshold` |double |25 |`--beamthreshold 15` |Cut of hypothesis far away by the current score from the best hypothesis |
|`lmweight` |double |0 |`--lmweight 1.1` |Language model weight to accumulate with acoustic model score |
|`wordscore` |double |0 |`--wordscore -0.2` |Score to add when word finishes (lexicon-based beam search decoder only) |
|`eosscore` |double |0 |`--eosscore 0.5` |Score to add when end of sentence is generated (for Seq2Seq criterion) |
|`silscore` |double |0 |`--silscore 0.5` |Silence appearance score to add (for CTC/ASG models) |
|`unkscore` |double |`-infinity` |`--unkscore 0.5` |Unknown word appearance score to add (CTC/ASG with lexicon-based beam-search decoder) |
|`smearing` |string: `none` / `max` / `logadd` |`none` |`--smearing none` |Smearing procedure in case of lexicon-based beam-search decoder only |

### Template to run beam-search decoder

We assume that saved `datadir`, `tokens` are stored with existing paths inside the AM model (otherwise you should redefine them in the command line command). Also `criterion`, `wordseparator` and `usewordpiece` will be loaded from the model. To use saved previously **Emission Set** exchange `am` flag to the `emission_dir`

```bash
<build/bin/fl_asr_decode> \
  --am=path/to/train/am.bin \
  --test=path/to/test/list/file \
  --maxload=10 \
  --nthread_decoder=2 \
  --show \
  --showletters \
  --lexicon=path/to/the/lexicon/file \
  --uselexicon=[true, false] \
  --lm=path/to/lm/file \
  --lmtype=[kenlm, convlm] \
  --decodertype=[wrd, tkn] \
  --beamsize=100 \
  --beamsizetoken=100 \
  --beamthreashould=20 \
  --lmweight=1 \
  --wordscore=0 \
  --eosscore=0 \
  --silscore=0 \
  --unkscore=0 \
  --smearing=max
```

#### Configuration file support

One can simply put all the flags into file, for example (name of the file `decode.cfg`)

```bash
--am=path/to/train/am.bin
--test=/absolute/path/to/test/list/file
--maxload=10
--nthread_decoder=2
--show
--sholetters
--lexicon=path/to/the/lexicon/file
--uselexicon=true
--lm=path/to/lm/file
--lmtype=kenlm
--decodertype=wrd
--beamsize=100
--beamsizetoken=100
--beamthreshold=20
```

and then run **Decode binary** with these flags (also one can add other flags in the command line)

```bash
<build/bin/fl_asr_decode> \
  --flagsfile=decode.cfg \
  --lmweight=1 \
  --wordscore=0 \
  --eosscore=0 \
  --silscore=0 \
  --unkscore=0 \
  --smearing=max
```
