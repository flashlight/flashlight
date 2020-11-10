Flashlight Python Bindings
==========================

flashlight Python binding supports for now part of the ``lib``:

- featurization of raw audio (MFCC, MFSC, etc.)
- ASG criterion (CUDA and CPU backends)
- Beam-search decoder (lexicon and lexicon free for CTC/ASG models with zerolm/kenlm language models)

Installation
************

Dependencies
############

We require ``python >= 3.6`` with the following packages installed:

- [packaging](https://pypi.org/project/packaging/)
- [torch](https://pypi.org/project/torch/)

[Anaconda](https://www.anaconda.com/distribution/) makes this easy. There are plenty of tutorials on how to set this up.

Aside from the above, the dependencies for Python bindings are a **strict subset** of the dependencies for the main flashlight build.
So if you already have the dependencies to build flashlight, you're all set to build python bindings as well.

The following dependencies **are required** to build python bindings:

- [KenLM](https://github.com/kpu/kenlm)
- [ATLAS](http://math-atlas.sourceforge.net/) or [OpenBLAS](https://www.openblas.net/)
- [FFTW3](http://www.fftw.org/)
- [cmake](https://cmake.org/) >= 3.5.1, and ``make``
- [CUDA](https://developer.nvidia.com/cuda-downloads) >= 9.2

Please refer to the flashlight dependencies sections for details on how to install the above packages.

The following dependencies **are not required** to build python bindings:

- flashlight itself
- libsndfile
- gflags
- glog

Build Instructions
##################

Once the dependencies are satisfied, simply run from wav2letter root:

.. code-block:: shell

    export KENLM_ROOT_DIR=<path/to>/kenlm
    cd bindings/python
    pip install -e .

Note that if you encounter errors, you'll probably have to ``rm -rf build`` before retrying the install.

Advanced Options
################

The following environment variables can be used to control various options:

- ``USE_CUDA=0`` removes the CUDA dependency, but you won't be able to use ASG criterion with CUDA tensors.
- ``USE_KENLM=0`` removes the KenLM dependency, but you won't be able to use the decoder unless you write C++ pybind11 bindings for your own LM.
- ``USE_MKL=1`` will use Intel MKL for featurization but this may cause dynamic loading conflicts.
- If you do not have ``torch``, you'll only have a raw pointer interface to ASG criterion instead of ``class ASGLoss(torch.nn.Module)``.

Build inside docker container
#############################

- For CUDA backend inside docker container run commands

  .. code-block:: shell

    pip install torch==1.2.0 packaging==19.1
    export KENLM_ROOT_DIR=/root/kenlm && \
    cd /root/flashlight/bindings/python && pip install -e .


- For CPU backend inside docker container run commands

  .. code-block:: shell

    pip install torch==1.2.0 packaging==19.1
    export USE_CUDA=0 && export KENLM_ROOT_DIR=/root/kenlm && \
    cd /root/flashlight/bindings/python && pip install -e .

Python API
************

Featurization
#############

Featurization module provides a bunch of classes for standard feature extraction from the audio data:
Ceplifter, Dct, Derivatives, Dither, Mfcc, Mfsc, PowerSpectrum, PreEmphasis, TriFilterbank, Windowing.

All of them have the method ``apply`` which can be used to transform the input data. For example:

.. code-block:: python

  # imports
  from flashlight.lib.audio.feature import FeatureParams, Mfcc
  import itertools as it

  # read the wave
  with open("path/to/file.wav") as f:
      wavinput = [float(x) for x in it.chain.from_iterable(line.split() for line in f)]

  # create params struct
  params = FeatureParams()
  params.sampling_freq = 16000
  params.low_freq_filterbank = 0
  params.high_freq_filterbank = 8000
  params.num_filterbank_chans = 20
  params.num_cepstral_coeffs = 13
  params.use_energy = False
  params.zero_mean_frame = False
  params.use_power = False

  # define transformation and apply to the wave
  mfcc = Mfcc(params)
  features = mfcc.apply(wavinput)


ASG Loss
########

ASG loss is a pytorch module (``nn.Module``) which supports CPU and CUDA backends.
It can be defined as

.. code-block:: python

    from flashlight.lib.sequence.criterion import ASGLoss
    asg_loss = ASGLoss(ntokens, scale_mode).to(device)


where ``ntokens`` is the number of tokens predicted for each frame (number of classes), ``scale_mode`` is a scaling factor which can be:

.. code-block:: python

    NONE = 0, # no scaling
    INPUT_SZ = 1, # scale to the input size
    INPUT_SZ_SQRT = 2, # scale to the sqrt of the input size
    TARGET_SZ = 3, # scale to the target size
    TARGET_SZ_SQRT = 4, # scale to the sqrt of the target size


Beam-search decoder
###################
Currently only lexicon-based beam-search decoder is supported. Also only n-gram (KenLM) language model is supported for python bindings.
However, one can define custom language model inside python and use it for decoding, details see below.
To have better understanding how this beam-search decoder works please see [Beam-search decoder section](TODO).

To run decoder one first should define its options:

.. code-block:: python

    from flashlight.lib.text.decoder import LexiconDecoderOptions, LexiconFreeDecoderOptions

    // for lexicon-based decoder
    options = LexiconDecoderOptions(
        beam_size, # number of top hypothesis to preserve at each decoding step
        token_beam_size, # restrict number of tokens by top am scores (if you have a huge token set)
        beam_threshold, # preserve a hypothesis only if its score is not far away from the current best hypothesis score
        lm_weight, # language model weight for LM score
        word_score, # score for words appearance in the transcription
        unk_score, # score for unknown word appearance in the transcription
        sil_score, # score for silence appearance in the transcription
        log_add, # the way how to combine scores during hypotheses merging (log add operation, max)
        criterion_type # supports only CriterionType.ASG or CriterionType.CTC
        )
    // for lexicon free-based decoder
    options = LexiconFreeDecoderOptions(
        beam_size, # number of top hypothesis to preserve at each decoding step
        token_beam_size, # restrict number of tokens by top am scores (if you have a huge token set)
        beam_threshold, # preserve a hypothesis only if its score is not far away from the current best hypothesis score
        lm_weight, # language model weight for LM score
        sil_score, # score for silence appearance in the transcription
        log_add, # the way how to combine scores during hypotheses merging (log add operation, max)
        criterion_type # supports only CriterionType.ASG or CriterionType.CTC
        )


Then we should prepare tokens dictionary (tokens for which acoustic models
returns probability for each frame), lexicon (mapping between words and its spelling with the tokens set).
Details on the tokens and lexicon files format have a look at
[Data Preparation](TODO).

.. code-block:: python

    from flashlight.lib.text.dictionary import Dictionary, load_words, create_word_dict


    tokens_dict = Dictionary("path/tokens.txt")
    # for ASG add used repetition symbols, for example
    # token_dict.add_entry("1")
    # token_dict.add_entry("2")

    lexicon = load_words("path/lexicon.txt") # returns LexiconMap
    word_dict = create_word_dict(lexicon) # returns Dictionary


To create language model for KenLM use

.. code-block:: python

    from flashlight.lib.text.decoder import KenLM


    lm = KenLM("path/lm.arpa", word_dict) # or "path/lm.bin"

Get the unknown and silence indices from the token dict and word dict to pass them into decoder:

.. code-block:: python

    sil_idx = token_dict.get_index("|")
    unk_idx = word_dict.get_index("<unk>")

Now define the lexicon ``Trie`` to restrict beam-search decoder search:

.. code-block:: python

    from flashlight.lib.text.decoder import Trie, SmearingMode
    from flashlight.lib.text.dictionary import pack_replabels


    trie = Trie(token_dict.index_size(), sil_idx)
    start_state = lm.start(False)

    def tkn_to_idx(spelling: list, token_dict : Dictionary, maxReps : int = 0):
        result = []
        for token in spelling:
            result.append(token_dict.get_index(token))
        return pack_replabels(result, token_dict, maxReps)


    for word, spellings in lexicon.items():
        usr_idx = word_dict.get_index(word)
        _, score = lm.score(start_state, usr_idx)
        for spelling in spellings:
            # convert spelling string into vector of indices
            spelling_idxs = tkn_to_idx(spelling, token_dict, 1)
            trie.insert(spelling_idxs, usr_idx, score)

        trie.smear(SmearingMode.MAX) # propagate word score to each spelling node to have some lm proxy score in each node.


Now we can run lexicon-based decoder:

.. code-block:: python

    import numpy
    from flashlight.lib.text.decoder import LexiconDecoder


    blank_idx = token_dict.get_index("#") # for CTC
    transitions = numpy.zeros((token_dict.index_size(), token_dict.index_size()) # for ASG fill up with correct values
    is_token_lm = False # we use word-level LM
    decoder = LexiconDecoder(options, trie, lm, sil_idx, blank_idx, unk_idx, transitions, is_token_lm)
    # emissions is numpy.array of acoustic model predictions with shape [T, N], where T is time, N is number of tokens
    results = decoder.decode(emissions.ctypes.data, T, N)
    # results[i].tokens contains tokens sequence (with length T)
    # results[i].score contains score of the hypothesis
    # results is sorted array with the best hypothesis stored with index=0.


Define your own language model for beam-search decoding
#######################################################

One can define custom language model in python and use it for beam-search decoding.

To deal with language model state we use the base class ``LMState`` and one can define additional
info corresponding to each state via creating ``dict(LMState, info)`` inside language model class:

.. code-block:: python

    import numpy
    from flashlight.lib.text.decoder import LM


    class MyPyLM(LM):
        mapping_states = dict() # store simple additional int for each state

        def __init__(self):
            LM.__init__(self)

        def start(self, start_with_nothing):
            state = LMState()
            self.mapping_states[state] = 0
            return state

        def score(self, state : LMState, token_index : int):
            """
            Evaluate language model based on the current lm state and new word
            Parameters:
            -----------
            state: current lm state
            token_index: index of the word
                        (can be lexicon index then you should store inside LM the
                        mapping between indices of lexicon and lm, or lm index of a word)

            Returns:
            --------
            (LMState, float): pair of (new state, score for the current word)
            """
            outstate = state.child(token_index)
            if outstate not in self.mapping_states:
                self.mapping_states[outstate] = self.mapping_states[state] + 1
            return (outstate, -numpy.random.random())

        def finish(self, state: LMState):
            """
            Evaluate eos for language model based on the current lm state

            Returns:
            --------
            (LMState, float): pair of (new state, score for the current word)
            """
            outstate = state.child(-1)
            if outstate not in self.mapping_states:
                self.mapping_states[outstate] = self.mapping_states[state] + 1
            return (outstate, -1)

LMState is a C++ base class for language model state. Its method ``compare`` (compare one state with another) is used inside the beam-search decoder.
It also has method ``LMState child(int index)`` returning a state which we obtained by following token with this index from current state.
Thus all states are organized as a trie.
We use ``child`` method in python to create this trie in a correct way (will be used inside decoder to compare states)
and then we can store additional info about state inside ``mapping_states``.

This language model can be used as (also printing the state and its additional stored info inside ``lm.mapping_states``):

.. code-block:: python

    custom_lm = MyLM()

    state = custom_lm.start(True)
    print(state, custom_lm.mapping_states[state])

    for i in range(5):
        state, score = custom_lm.score(state, i)
        print(state, custom_lm.mapping_states[state], score)

    state, score = custom_lm.finish(state)
    print(state, custom_lm.mapping_states[state], score)

and for decoder:

.. code-block:: python

    decoder = LexiconDecoder(options, trie, custom_lm, sil_idx, blank_inx, unk_idx, transitions, False)


Examples
********

After flashlight python package is installed, please, have a look at the examples how to use classes and methods of flashlight from python.

- ASG criterion

  .. code-block:: shell

    # with cpu backend
    python example/criterion_example.py --cpu
    # with gpu backend
    python example/criterion_example.py


- lexicon beam-search decoder with KenLM word-level language model

  .. code-block:: shell

    python example/decoder_example.py ../../flashlight/app/asr/test/decoder/data

- audio featurization

  .. code-block:: shell

    python example/feature_example.py ../../flashlight/lib/test/audio/feature/data
