        # --obj_path scene0002_00 \
        # --label 2 \
        # --prompt "brick floor" \
        # --image "./image_prompt/brick_floor.jpg" \
        # --output_dir results/test/scene0002_00/image_text_prompt/brick_floor_iter3000 \


CUDA_VISIBLE_DEVICES=3 python -m pdb sem_seg_main.py \
\
        --obj_path scene0055_00 \
        --label 14 2 4 \
        --prompt "wooden desk, marble floor, steel bed" \
        --output_dir results/augs_comp/scene0055_00/n_1_a_1 \
        --forbidden "human face,English alphabet,lighting,human" \
\
        --learning_rate 0.001 \
        --lr_decay 0.9 \
        --n_iter 1000 \
\
        --frontview_elev_std 0.01 \
        --frontview_azim_std 0.1 \
        --background 0.1 0.1 0.1 \
        --render_all_grad_one \
\
        --n_normaugs 1 \
        --n_augs 1 \
        --n_views 4 \
        --mincrop 0.75 \
        --maxcrop 1.0 \
        --view_min 0.2 \
        --view_max 0.8 \
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
