# Language Modeling

## Build Dictionary

```
fl_lm_dictionary_builder \
 --data_dir=/tmp \
 --data_train=test1.txt,test2.txt \
 --n_workers=40 \
 --dictionary=dictionary.txt \
 --dictionary_min_appearence=2 \
 --dictionary_max_size=200000 \
 --write_meta=true
```

Dictionary builder reads all the text files specified in `--data_train` from `--data_dir` and count the total number of tokens and sentences in it, using `--n_workers` threads in parallel. After filtering out the uncommon tokens with `--dictionary_min_appearence` and limiting the dictionary size by `--dictionary_max_size`, dictionary builder will save out a dictionary with tokens and their number of appearance in all the text files into `--dictionary`. If `--write_meta` is on, the meta data of each text file will be generated in `--data_dir` with suffix `.desc`. Meta data describes the beginning position (in byte) of each sentence and the number of tokens in it.

Built dictionary will also contain special tokens at the beginning, so you don't need to tweak this dictionary before training.
- `</s>` - end of sentence
- `<unk>` - unknown token
- `<pad>` - pad token
- `<mask>` - mask token (is needed for BERT training)

## Train

### Compile the model plugin

Add a compiler flag `-DFL_PLUGIN_MODULE_SRC_PATH` to the cmake command pointing to the model you want to use. For example, `-DFL_PLUGIN_MODULE_SRC_PATH=../flashlight/app/lm/plugins/LmAdae512SinposL8H8Fc1024Dp03Ldp0Adsm.cpp`. With the dynamic library created for the model, you can simply pass it into the training binary like `--train_arch_file=LmAdae512SinposL8H8Fc1024Dp03Ldp0Adsm.so`.

### Training modes
- `train`: Train a model from scratch, and save logs and checkpoints into `exp_rundir/exp_model_name`.
- `continue`: Continue training an existing model in `exp_rundir/exp_model_name`.
- `fork`: Training a new model with weights initialized to the one specified in `--FLAGS_exp_init_model_path`.

### Training tasks
- Auto-regressive training (`--train_task=autoreg`)
```
fl_lm_train \
  --exp_rundir=/tmp \
  --exp_model_name=my_lm \
  --data_dir=<...>/my_data \
  --data_valid=data1.txt,data2.txt \
  --data_sample_break_mode=none \
  --data_tokens_per_sample=4096 \
  --dictionary=<...>/my_dict.txt \
  --loss_type=ce \
  --train_arch_file=/path/to/compiled/myarch.so \
  --train_max_grad_norm=0.1 \
  --train_report_updates=1000 \
  --train_save_updates=13000 \
  --train_warmup_init_lr=1e-7 \
  --train_optimizer=nag \
  --train_lr=1 \
  --train_lr_schedule=invsqrt
  --train_momentum=0.9 、
  --train_weight_decay=0
```

- BERT-style training (`--train_task=mask`)
```
fl_lm_train \
  --exp_rundir=/tmp \
  --exp_model_name=my_lm \
  --data_dir=<...>/my_data \
  --data_valid=data1.txt,data2.txt \
  --data_sample_break_mode=none \
  --data_tokens_per_sample=4096 \
  --dictionary=<...>/my_dict.txt \
  --loss_type=adsm \
  --loss_adsm_input_size=1024 \
  --loss_adsm_cutoffs=10000,50000,150000 \
  --train_task=mask
  --train_arch_file=/path/to/compiled/myarch.so \
  --train_max_grad_norm=0.1 \
  --train_report_updates=1000 \
  --train_save_updates=13000 \
  --train_warmup_init_lr=1e-7 \
  --train_optimizer=nag \
  --train_lr=1 \
  --train_lr_schedule=invsqrt
  --train_momentum=0.9 、
  --train_weight_decay=0
  --mask_rand_token_prob=0.1 \
  --mask_same_token_prob=0.1 \
  --mask_prob=0.15
```

A complete list of the flag definitions and short descriptions of their meaning can be found [here](https://github.com/flashlight/flashlight/blob/master/flashlight/app/lm/Trainer.cpp).


## Evaluation

To evaluate a model, simple run
```
fl_lm_test \
  --train_arch_file=/path/to/your/arch.so \
  --exp_init_model_path=/path/to/your/model.bin \
  --dictionary=/path/to/your/dict.txt \
  --dictionary_max_size=200000 \
  --data_dir=/path/to/your_data \
  --data_valid=data.txt
```
