        # --obj_path scene0137_02 \
        # --output_dir results/demo/comparison/scene0137_02/with_initial_color \
        # --prompt "brick wall, marble floor, wooden chairs, leather bed" \
        # --label 1 2 5 4 \
        # --obj_path scene0686_01 \
        # --output_dir results/demo/comparison/scene0686_01/init \
        # --prompt "icy wall, steel floor, Golden toilet, wooden sink" \
        # --label 1 2 33 34 \
        # --obj_path scene0355_00 \
        # --output_dir results/demo/comparison/scene0355_00/test \
        # --prompt "Bamboo wall, brick floor, plastic chair, stone table" \
        # --label 1 2 5 7 \
        # --obj_path scene0449_00 \
        # --output_dir results/demo/comparison/scene0449_00/init \
        # --prompt "cement floor, fabric towel, wooden door, steel sink, golden toilet, " \
        # --label 2 27 8 34 33 \
        # --obj_path scene0435_01 \
        # --output_dir results/demo/comparison/scene0435_01/init \
        # --prompt "Diamond bed, icy floor, wall with murals, leather pillow, marble desk" \
        # --label 4 2 1 18 14 \
        # --obj_path scene0434_01 \
        # --output_dir results/demo/comparison/scene0434_01/init \
        # --prompt "diamond toilet, marble wall, wooden cabinet, steel bathtub, shower curtain with rainbow" \
        # --label 33 1 3 36 28 \
        # --obj_path scene0422_00 \
        # --output_dir results/demo/comparison/scene0422_00/init \
        # --prompt "chocolate wall, ocean floor, glass table, steel cabinet, leather chair" \
        # --label 1 2 7 3 5 \

CUDA_VISIBLE_DEVICES=2 python sem_seg_main.py \
        --run branch \
        --obj_path scene0422_00 \
        --output_dir results/demo/comparison/scene0422_00/init \
        --prompt "icy wall, ocean floor, wooden chair, water curtain, " \
        --label 1 2 5 16 \
        --sigma 5.0  \
        --clamp tanh \
        --n_normaugs 4 \
        --n_augs 1 \
        --normmincrop 0.1 \
        --normmaxcrop 0.1 \
        --geoloss \
        --colordepth 2 \
        --normdepth 2 \
        --frontview \
        --frontview_std 4 \
        --clipavg view \
        --lr_decay 0.9 \
        --clamp tanh \
        --normclamp tanh \
        --maxcrop 1.0 \
        --save_render \
        --seed 11 \
        --n_iter 1000 \
        --learning_rate 0.0005 \
        --normal_learning_rate 0.0005 \
        --background 0.5 0.5 0.5 \
        --rand_background \
        --frontview_center 1.8707 0.6303 \
        --lighting \
        --normratio 0.05 \
        --color_only \
        --render_all_grad_one \
        --focus_one_thing