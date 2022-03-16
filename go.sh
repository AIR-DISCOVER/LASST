python sem_seg_main.py \
\
        --obj_path scene0431_00 \
        --label 1 \
        --prompt "wooden wall" \
        --forbidden "human face" \
        --output_dir results/test/scene0158_02 \
\
        --learning_rate 0.001 \
        --lr_decay 0.9 \
        --n_iter 1000 \
\
        --frontview_elev_std 12 \
        --frontview_azim_std 6 \
        --background 0.5 0.5 0.5 \
        --rand_background \
        --with_prior_color \
        --render_one_grad_one \
\
        --n_normaugs 4 \
        --n_augs 1 \
        --mincrop 0.5 \
        --maxcrop 1.0 \
        --normmincrop 0.3 \
        --normmaxcrop 0.9 \
\
        --color_only \
\
        --clipavg \
        --sv_stat_loss_weight 0.1 \
        --report_step 10 
        
        # --hsv_stat_loss_weight 0.5 \
        # --hsv_loss_weight 0.01 \
        
        #--regress \
