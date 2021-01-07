#!/bin/bash
set -x
# Be very wary of this explicit setting of CUDA_VISIBLE_DEVICES. Say you are
# running one task and asked for gres=gpu:1 then setting this variable will mean
# all your processes will want to run GPU 0 - disaster!! Setting this variable
# only makes sense in specific cases that I have described above where you are
# using gres=gpu:8 and I have spawned 8 tasks. So I need to divvy up the GPUs
# between the tasks. Think THRICE before you set this!!
#export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
echo $SLURM_NTASKS
echo $SLURM_LOCALID

# Needed for arrayfire
export LD_LIBRARY_PATH=/private/home/padentomasello/usr/lib/:$LD_LIBRARY_PATH

BUILD_DIR=/scratch/slurm_tmpdir/$SLURM_JOB_ID/$1
EVAL_DIR=$BUILD_DIR/eval/$SLURM_LOCALID/
RUN_DIR=/checkpoint/padentomasello/models/$SLURM_JOB_ID/
mkdir -p $RUN_DIR
mkdir -p $BUILD_DIR/rndv/
mkdir -p $EVAL_DIR
$BUILD_DIR/flashlight/build/Detr train -lr 0.0001 --epochs 500 --batch_size 2 \
--world_rank $SLURM_LOCALID --world_size $SLURM_NTASKS \
--rndv_filepath $BUILD_DIR/rndv/ \
--eval_dir $EVAL_DIR \
--tryfromenv=eval_iters,data_dir,metric_iters,pretrained,print_params  \
--pytorch_init /checkpoint/padentomasello/models/detr/pytorch_initializaition_dropout \
--rundir $RUN_DIR
2>&1 # Ugh why does FL log send to std::err? 



# Your CUDA enabled program here
