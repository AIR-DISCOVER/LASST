        # --obj_path scene0002_00 \
        # --label 2 \
        # --prompt "brick floor" \
        # --image "./image_prompt/brick_floor.jpg" \
        # --output_dir results/test/scene0002_00/image_text_prompt/brick_floor_iter3000 \


CUDA_VISIBLE_DEVICES=0 python sem_seg_main.py \
\
        --obj_path scene0002_00 \
        --label 2 \
        --prompt "brick floor" \
        --output_dir results/augs_comp/scene0002_00/n_3_a_4 \
\
        --learning_rate 0.0005 \
        --lr_decay 0.9 \
        --n_iter 1000 \
\
        --frontview_center 0.4 0.2 \
        --frontview_std 4 \
        --background 0.5 0.5 0.5 \
        --rand_background \
        --render_one_grad_one \
\
        --n_normaugs 3 \
        --n_augs 4 \
        --maxcrop 1.0 \
        --normmincrop 0.1 \
        --normmaxcrop 0.1 \
\
        --color_only\
        --with_prior_color\
\
        --hsv_loss_weight 0.1 \
        --report_step 100
