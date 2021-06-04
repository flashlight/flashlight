# Baselines
python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=dev

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline_run2.cfg --mode=train --ngpu=16 --partition=dev

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline_noaug.cfg --mode=train --ngpu=16 --partition=learnfair

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline_noaug_correct.cfg --mode=train --ngpu=16 --partition=learnfair

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=dev --extra="--image_size=160"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=32 --partition=learnfair --extra="--image_size=384 --data_batch_size=32"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=32 --partition=learnlab --extra="--image_size=384 --data_batch_size=32 --train_seed=42"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=learnlab --extra="--posemb_dropout=0.1"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=learnlab --extra="--posemb_dropout=0.3"

# Eval 
/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 2 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr

# No pos emb

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=learnfair --extra="--use_pos_emb=false"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=learnfair --extra="--use_pos_emb=false --train_seed=42"

# Use sin2d but no aug

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=learnfair --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=false"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=learnfair --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=false --train_seed=42"

## with 10 as base
python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=devlab --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=false --train_seed=47" --suffix="base10"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=learnlab --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=false --train_seed=47 --image_size=160" --suffix="base10"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=32 --partition=learnlab --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=false --train_seed=47 --image_size=384 --data_batch_size=32" --suffix="base10"

# Use sin2d + aug

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=dev --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=learnlab --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true --image_size=160"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=32 --partition=learnlab --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true --image_size=384 --data_batch_size=32"

## with 10 as base
python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=devlab --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true --train_seed=47" --suffix="base10"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=learnlab --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true --train_seed=47 --image_size=160" --suffix="base10"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=32 --partition=learnlab --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true --train_seed=47 --image_size=384 --data_batch_size=32" --suffix="base10"

# Use sin2d + aug no shrink

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=dev --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true --do_shrink=false"

## with base 10
python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=devlab --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true --train_seed=47 --do_shrink=false" --suffix="base10"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=learnlab --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true --train_seed=47 --do_shrink=false --image_size=160" --suffix="base10"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=32 --partition=learnlab --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true --train_seed=47 --do_shrink=false --image_size=384 --data_batch_size=32" --suffix="base10"

## no training aug

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline.cfg --mode=train --ngpu=16 --partition=learnfair --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true --noaug=true --train_aug_use_mix=false"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/baseline_noaug_correct.cfg --mode=train --ngpu=16 --partition=learnfair --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true"

## CAPE different sizes training
#python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/cape_randsize_384.cfg --mode=train --ngpu=32 --partition=learnfair 

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/cape_randsize_224.cfg --mode=train --ngpu=16 --partition=learnfair

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/cape_randsize_224.cfg --mode=train --ngpu=16 --partition=learnfair --extra="--random_size_min=128"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/cape_randsize_224_newbin.cfg --mode=train --ngpu=32 --partition=learnfair


python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/cape_randsize_224_newbin.cfg --mode=train --ngpu=8 --partition=dev --extra="--do_shrink=false"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/cape_randsize_224_newbin.cfg --mode=train --ngpu=8 --partition=dev --extra="--do_shrink=false --dyn_batch=true"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/cape_randsize_224_newbin.cfg --mode=train --ngpu=8 --partition=dev --extra="--do_shrink=true --dyn_batch=true"

## MNIST

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline.cfg --mode=train --ngpu=2 --partition=learnfair

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline.cfg --mode=train --ngpu=2 --partition=learnfair --extra="--use_pos_emb=false"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline.cfg --mode=train --ngpu=2 --partition=learnfair --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=false"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline.cfg --mode=train --ngpu=2 --partition=dev --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline_336.cfg --mode=train --ngpu=8 --partition=dev --extra="--use_pos_emb=false"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline_336.cfg --mode=train --ngpu=8 --partition=dev --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline_random.cfg --mode=train --ngpu=2 --partition=learnfair --extra="--use_pos_emb=false"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline_random.cfg --mode=train --ngpu=2 --partition=learnfair --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true"



python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline_random_224.cfg --mode=train --ngpu=2 --partition=dev

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline_random_224.cfg --mode=train --ngpu=2 --partition=dev --extra="--use_pos_emb=false"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline_random_224.cfg --mode=train --ngpu=2 --partition=dev --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=false"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline_random_224.cfg --mode=train --ngpu=2 --partition=dev --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline_random_permute_224.cfg --mode=train --ngpu=2 --partition=dev

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline_random_permute_224.cfg --mode=train --ngpu=2 --partition=dev --extra="--use_pos_emb=false"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline_random_permute_224.cfg --mode=train --ngpu=2 --partition=dev --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=false"

python3 /checkpoint/antares/experiments/fl_new_master/augposemb/vit/train_vision_tr.py --binary=/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_vit --config=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/mnist_baseline_random_permute_224.cfg --mode=train --ngpu=2 --partition=dev --extra="--use_pos_emb=false --use_aug_pos_emb=true --use_aug_pos_emb_aug=true"


