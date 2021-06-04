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

/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 8 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/$1/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=160 --data_batch_size=256

/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 8 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/$1/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=224 --data_batch_size=256

/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 8 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/$1/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=384 --data_batch_size=128

/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 8 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/$1/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=672 --data_batch_size=20

for data in "imagenetv2-threshold0.7-format-val" "imagenetv2-matched-frequency-format-val" "imagenetv2-top-images-format-val"
do
	/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 8 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/$1/model --data_dir=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/imagenet_v2/$data --logtostderr --image_size=160 --data_batch_size=256

	/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 8 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/$1/model --data_dir=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/imagenet_v2/$data --logtostderr --image_size=224 --data_batch_size=256

	/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 8 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/$1/model --data_dir=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/imagenet_v2/$data --logtostderr --image_size=384 --data_batch_size=128

	/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 8 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/$1/model --data_dir=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/imagenet_v2/$data --logtostderr --image_size=672 --data_batch_size=20
done
