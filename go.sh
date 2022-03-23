CUDA_VISIBLE_DEVICES=1 python sem_seg_main.py \
\
        --obj_path scene0002_00 \
        --label 6 \
        --prompt "leather sofa" \
        --forbidden "human face,English alphabet,lighting,human" \
        --output_dir results/test/scene0002_00/new_3_7 \
\
        --learning_rate 0.001 \
        --lr_decay 0.9 \
        --n_iter 1000 \
\
        --frontview_elev_std 0.01 \
        --frontview_azim_std 0.1 \
        --background 0.1 0.1 0.1 \
        --render_one_grad_one \
\
        --n_normaugs 3 \
        --n_augs 3 \
        --n_views 4 \
        --mincrop 0.6 \
        --maxcrop 0.9 \
        --view_min 0.3 \
        --view_max 0.7 \
        --normmincrop 0.6 \
        --normmaxcrop 0.9 \
\
        --color_only\
        --with_prior_color\
\
        --clipavg \
        --sv_stat_loss_weight 0.1 \
        --report_step 100 


        # --image star.jpg \
        # --fixed_all \
        # --rand_background \
        # --hsv_stat_loss_weight 0.5 \
        # --hsv_loss_weight 0.01 \
        
        #--regress \
