python sem_seg_main.py \
\
        --obj_path scene0431_00 \
        --label 3 \
        --prompt "Van Gogh cabinet" \
        --forbidden "human face,English alphabet,lighting,human" \
        --output_dir results/test/scene0158_02 \
\
        --learning_rate 0.001 \
        --lr_decay 0.9 \
        --n_iter 1000 \
\
        --frontview_elev_std 0.01 \
        --frontview_azim_std 0.1 \
        --background 0.1 0.1 0.1 \
        --with_prior_color \
        --render_one_grad_one \
\
        --n_normaugs 4 \
        --n_augs 1 \
        --n_views 8 \
        --mincrop 0.75 \
        --maxcrop 1.0 \
        --view_min 0.2 \
        --view_max 0.8 \
        --normmincrop 0.6 \
        --normmaxcrop 0.9 \
\
        --color_only \
\
        --clipavg \
        --sv_stat_loss_weight 0.1 \
        --report_step 10 
        
        # --image star.jpg \
        # --fixed_all \
        # --rand_background \
        # --hsv_stat_loss_weight 0.5 \
        # --hsv_loss_weight 0.01 \
        
        #--regress \
