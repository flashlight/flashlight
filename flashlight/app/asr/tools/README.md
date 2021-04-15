# Tools

This directory contains tools for audio analysis and processing built on flashlight.

To build the tools, ensure `-DFL_BUILD_APP_ASR_TOOLS=ON` as a CMake flag when building the ASR app.

<details>
<summary>Audio Force Alignment</summary>

See the [`alignment` readme](https://github.com/jacobkahn/flashlight/tree/export-D25647716/flashlight/app/asr/tools/alignment) for documentation.
</details>

<details>
<summary>Voice Activity Detection</summary>

## Voice Activity Detection with CTC + an n-gram Language Model
`VoiceActivityDetection-CTC` contains a simple pipeline that supports a CTC-trained acoustic model trained with Flashlight and n-gram language model in an accepted binary format (see the [decoder documentation](https://github.com/flashlight/flashlight/blob/master/flashlight/app/asr/README.md#beam-search-decoder) for more).

### Using the Pipeline
Build the tool with `make fl_asr_voice_activity_detection_ctc -j$(nproc)`.

#### Input List File
First, create an input list file containing the audio data. The list file should exactly follow the standard wav2letter [list input format for training](https://github.com/flashlight/flashlight/blob/master/flashlight/app/asr/README.md#audio-and-transcriptions-data), but the transcriptions column should be empty. For instance:
```
// Example input file

[~/speech/data] head analyze.lst
train001 /tmp/000000000.flac 100.03
train002 /tmp/000000001.flac 360.57
train003 /tmp/000000002.flac 123.53
train004 /tmp/000000003.flac 999.99
...
...
```

#### Running
Run the binary:
```
[path to binary]/fl_asr_voice_activity_detection_ctc \
    --am [path to model] \
    --lm [path to language model] \
    --test [path to list file] \
    --lexicon [path to lexicon file] \
    --maxload -1 \
    --datadir= \
    --tokens [path to tokens file] \
    --outpath [output directory]
```

The script outputs four files named by each input sample ID in the directory specified by outpath:
1. A `.vad` file containing chunk-level probabilities of non-speech based on the probability of silence. These are assigned for each chunk of output; for a model trained with a stride of 1, these will be each frame (10 ms), but for a model with a stride of 8, these will be (80 ms) chunks.
2. An `.sts` file containing the perplexity the predicted sequence based on a specified input in addition to the percentage of the audio containing speech based on the passed `--vadthreshold`.
3. A `.tsc` file containing the most likely token-level transcription of given audio based on the acoustic model output only.
4. A `.fwt` file containing frame or chunk-level token emissions based on the most-likely token emitted for each sample.

### Acoustic Models for Audio Analysis

Below are baseline models usable with the tool, although any model/lexicon/token set can be used..

| File | Dataset | Dev Set | Criterion | Architecture | Lexicon | Tokens |
| - | - | - | - | - | - | - |
| [baseline_dev-other](https://dl.fbaipublicfiles.com/wav2letter/audio_analysis/tds_ctc/model.bin) | LibriSpeech | dev-other | CTC | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/audio_analysis/tds_ctc/arch.txt) | [Lexicon](https://dl.fbaipublicfiles.com/wav2letter/audio_analysis/tds_ctc/dict.lst) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/audio_analysis/tds_ctc/tokens.lst) |

</details>
