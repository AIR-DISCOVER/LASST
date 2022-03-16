python sem_seg_main.py \
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
\
        --hsv_loss_weight 10 \
        --report_step 10 \
        --dry_run
        
        
        # --clipavg \
        #--regress \
