#!/bin/bash
set -e
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
RUN_DIR=/checkpoint/padentomasello/models/$2/
mkdir -p $RUN_DIR
mkdir -p $BUILD_DIR/rndv/
mkdir -p $EVAL_DIR
$BUILD_DIR/flashlight/build/Detr continue $RUN_DIR \  # Ugh why does FL log send to std::err? 
--eval_dir $EVAL_DIR \
--eval_script $BUILD_DIR/flashlight/flashlight/app/objdet/scripts/eval_coco.py \
 2>&1



# Your CUDA enabled program here
