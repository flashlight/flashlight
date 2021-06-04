#!/bin/bash

#SBATCH --cpus-per-task=10
#SBATCH --partition=devlab,learnlab
#SBATCH --gres=gpu:volta:1
#SBATCH --constraint=volta16gb
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --mem=30GB
#SBATCH --signal=B:USR1@200
#SBATCH --comment=Neurips
#SBATCH --open-mode=append
#SBATCH --job-name=vit:eval
#SBATCH --output=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/log/%j.out
#SBATCH --error=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/log/%j.err

# 1. Load modules
module purge
module load cuda/11.0
module load cudnn/v8.0.3.33-cuda.11.0
module load NCCL/2.8.3-1-cuda.11.0
module load intel/mkl/2020.3.279
module load kenlm/010421/gcc.9.3.0
export NCCL_SOCKET_IFNAME=^docker0,lo


/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=96 --data_batch_size=24000 --distributed_enable=false
/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=96 --data_batch_size=24300 --distributed_enable=false --speed

/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=128 --data_batch_size=12000 --distributed_enable=false
/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=128 --data_batch_size=12100 --distributed_enable=false --speed

/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=160 --data_batch_size=7600 --distributed_enable=false
/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=160 --data_batch_size=8600 --distributed_enable=false --speed

/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=192 --data_batch_size=4000 --distributed_enable=false
/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=192 --data_batch_size=5000 --distributed_enable=false --speed

/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=224 --data_batch_size=2500 --distributed_enable=false
/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=224 --data_batch_size=3650 --distributed_enable=false --speed

/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=256 --data_batch_size=2100 --distributed_enable=false
/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=256 --data_batch_size=2700 --distributed_enable=false --speed

/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=288 --data_batch_size=1700 --distributed_enable=false
/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=288 --data_batch_size=1950 --distributed_enable=false --speed

/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=320 --data_batch_size=1150 --distributed_enable=false
/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=320 --data_batch_size=1450 --distributed_enable=false --speed

/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=352 --data_batch_size=700 --distributed_enable=false
/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=352 --data_batch_size=1130 --distributed_enable=false --speed

/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=384 --data_batch_size=700 --distributed_enable=false
/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_vitt_s12_noregaug/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=384 --data_batch_size=800 --distributed_enable=false --speed


