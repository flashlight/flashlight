#!/bin/bash

#SBATCH --cpus-per-task=10
#SBATCH --partition=devlab,learnlab
#SBATCH --gres=gpu:volta:8
#SBATCH --constraint=volta32gb,bldg2
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=440GB
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

/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 8 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_exp10_s12/model --data_dir=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/imagenet_bins --logtostderr --image_size=160 --data_batch_size=256 --data_path=$1

/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 8 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_exp10_s12/model --data_dir=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/imagenet_bins --logtostderr --image_size=224 --data_batch_size=256 --data_path=$1

/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 8 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_exp10_s12/model --data_dir=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/imagenet_bins --logtostderr --image_size=384 --data_batch_size=128 --data_path=$1

/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 8 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_exp10_s12/model --data_dir=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/imagenet_bins --logtostderr --image_size=$2 --data_batch_size=256 --data_path=$1

/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 8 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_exp10_s12/model --data_dir=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/imagenet_bins --logtostderr --image_size=$3 --data_batch_size=256 --data_path=$1

/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 8 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_exp10_s12/model --data_dir=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/imagenet_bins --logtostderr --data_batch_size=1 --data_path=$1 --use_own_size=true

/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 8 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_exp10_s12/model --data_dir=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/imagenet_bins --logtostderr --data_batch_size=1 --data_path=$1 --use_own_size=true --nocrop=true
