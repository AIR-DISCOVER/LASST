IDX=$RANDOM
echo $IDX
python sem_seg_main.py \
        --obj_path scene0158_02 \
        --output_dir results/test/$IDX/scene0158_02 \
        --prompt "marble wall" \
        --label 1 \
        --sigma 5.0  \
        --clamp tanh \
        --n_normaugs 4 \
        --n_augs 1 \
        --normmincrop 0.1 \
        --normmaxcrop 0.1 \
        --colordepth 2 \
        --normdepth 2   \
        --frontview_std 4 \
        --clipavg view \
        --lr_decay 0.9 \
        --clamp tanh \
        --normclamp tanh  \
        --maxcrop 1.0 \
        --save_render \
        --seed 42 \
        --n_iter 100 \
        --learning_rate 0.0005 \
        --background 0.5 0.5 0.5 \
        --rand_background \
        --frontview_center 1.7 0.75 \
        --with_prior_color \
        --normratio 0.05 \
        --color_only \
        --render_one_grad_one \
        --with_hsv_loss \
        --report_step 10
        #--regress \
echo $IDX