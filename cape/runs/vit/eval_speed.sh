for s in 64 96 128 160 192 224 256 288 320 352 384 
do
/checkpoint/antares/experiments/fl_new_master/fl_tr_trick/build/bin/imgclass/fl_img_imagenet_eval --exp_checkpoint_path=/checkpoint/qiantong/fl_experiments/vision_tr/cape_mixsize_exp10_s12/model --data_dir=/datasets01/imagenet_full_size/061417 --logtostderr --image_size=$s --data_batch_size=128 --distributed_enable=false
done
