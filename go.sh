IDX=$RANDOM
echo $IDX
python sem_seg_main.py \
        --run branch \
        --obj_path scene0158_02 \
        --output_dir results/teaser/scene0158_02 \
        --prompt "ceramic refridgerator,marble wall,diamond texture floor,wooden cabinet,diamond texture door,steel counter," \
        --label 24 1 2 3 8 12 \
        --sigma 5.0  \
        --clamp tanh \
        --n_normaugs 4 \
        --n_augs 1 \
        --normmincrop 0.1 \
        --normmaxcrop 0.1 \
        --colordepth 2 \
        --normdepth 2   \
        --frontview \
        --frontview_std 4 \
        --clipavg view \
        --lr_decay 0.9 \
        --clamp tanh \
        --normclamp tanh  \
        --maxcrop 1.0 \
        --save_render \
        --seed 11 \
        --n_iter 1500 \
        --learning_rate 0.0005 \
        --normal_learning_rate 0.0005 \
        --background 0.5 0.5 0.5 \
        --rand_background \
        --frontview_center 1.8707 0.6303 \
        --with_prior_color \
        --normratio 0.05 \
        --color_only \
        --focus_one_thing
echo $IDX