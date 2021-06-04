for data in "imagenetv2-threshold0.7-format-val" "imagenetv2-matched-frequency-format-val" "imagenetv2-top-images-format-val" 
do
/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 2 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/$1/model --data_dir=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/imagenet_v2/$data --logtostderr --image_size=160 --data_batch_size=256

/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 2 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/$1/model --data_dir=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/imagenet_v2/$data --logtostderr --image_size=224 --data_batch_size=256

/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 2 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/$1/model --data_dir=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/imagenet_v2/$data --logtostderr --image_size=384 --data_batch_size=128

/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun -n 2 /checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/$1/model --data_dir=/checkpoint/antares/experiments/fl_new_master/augposemb/vit/imagenet_v2/$data --logtostderr --image_size=672 --data_batch_size=20
done
