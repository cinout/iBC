python -u main_train.py \
    --method mocov2 \
    --trigger_type htba \
    \
    --image_size 64 \
    \
    --magnitude_train 300.0 \
    --magnitude_val 300.0 \
    --dataset imagenet100 \
    --target_class 26 \
    \
    --ssl_pretrain_seed 42 \
    --num_views 64 \
    \
    --use_trigger_channel_removal \
    --find_channels_from_n_poison_samples 4 \
    --find_channels_from_n_clean_samples 0 \
    --voted_channel_num 30 \
    --removed_channel_num 70 \
    \
    --note "#poi=4, vote=30 rm=70, replace_value=0.4" \
    --pretrained_ssl_model "./Experiments/imagenet100_htba_mocov2_ref_set_sd42/encoder.pth.tar" \
    --pretrained_linear_model "./Experiments/imagenet100_htba_mocov2_ref_set_sd42/linear.pth.tar" \