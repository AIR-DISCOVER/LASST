python sem_seg_main.py \
<<<<<<< HEAD
\
        --obj_path scene0158_02 \
        --label 1 \
        --prompt "marble wall" \
        --output_dir results/test/scene0158_02 \
\
        --learning_rate 0.0005 \
        --lr_decay 0.9 \
        --n_iter 1000 \
\
        --frontview_center 1.7 1.57 \
        --frontview_std 4 \
        --background 0.5 0.5 0.5 \
        --rand_background \
=======
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
>>>>>>> 2e0f8ee8180fe9f94121a5774991c3b4d82d9235
        --with_prior_color \
        --render_one_grad_one \
\
        --n_normaugs 4 \
        --n_augs 1 \
        --maxcrop 1.0 \
        --normmincrop 0.1 \
        --normmaxcrop 0.9 \
\
        --color_only \
<<<<<<< HEAD
\
        --hsv_loss_weight 10 \
        --report_step 10
        
        
        # --clipavg \
        #--regress \
=======
        --render_one_grad_one \
        --with_hsv_loss \
        --report_step 10
        #--regress \
echo $IDX
>>>>>>> 2e0f8ee8180fe9f94121a5774991c3b4d82d9235
