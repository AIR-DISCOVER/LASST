# %%
from random import shuffle, random, randint
from launch import launch
import sys
import os
from datetime import datetime

SCENE_LIST = os.listdir('/home/tb5zhh/data/full/train')

DATE = datetime.today().strftime('%Y-%m-%d')
# 2022-02-07: randomly choose: scene_id, label_id, texture, one class in one scene at a time e.g. "a rusted door"
# 2022-02-10: new textures, and keep only the textures as prompt e.g. "Nebula"
# 2022-02-12: adopt new textures and apply those to all classes in a scene
# 2022-03-01: fix the big bug of incorrespondance between class and label

COMMAND = f'python main.py \
        --run branch \
        --obj_path $SCENE_ID$ \
        --output_dir \"results/batch/{DATE}/$SCENE_ID$/$NAME$\" \
        --prompt \"$PROMPT$\" \
        --label {" ".join([str(i) for i in list(range(1, 37))])}\
        --sigma 5.0  \
        --clamp tanh \
        --n_normaugs 4 \
        --n_augs 1 \
        --normmincrop 0.1 \
        --normmaxcrop 0.1 \
        --cropforward \
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
        --focus_one_thing'

# %%

COLORS = [
    'ash grey',
    'eton blue',
    'celadon green',
    'mint green',
    'navy blue',
    'sapphire blue',
    'turquoise blue',
    'Van Dyke brown',
    'aero blue',
    'golden',
    'African violet',
]
TEXTURE = [
    'wooden',
    'marble',
    'plastic',
    'steel',
    'leather',
    'diamond texture',
    'fabric',
    'icy',
    'ceramic'
]

CLASS_LABELS = (
    (1, "wall"),
    (2, "floor"),
    (3, "cabinet"),
    (4, "bed"),
    (5, "chair"),
    (6, "sofa"),
    (7, "table"),
    (8, "door"),
    (9, "window"),
    (10, "bookshelf"),
    (11, "picture"),
    (12, "counter"),
    (13, "blinds"),
    (14, "desk"),
    (15, "shelves"),
    (16, "curtain"),
    (17, "dresser"),
    (18, "pillow"),
    (19, "mirror"),
    (20, "floor mat"),
    (21, "clothes"),
    (22, "ceiling"),
    (23, "books"),
    (24, "refridgerator"),
    (25, "television"),
    (26, "paper"),
    (27, "towel"),
    (28, "shower curtain"),
    (29, "box"),
    (30, "whiteboard"),
    (31, "person"),
    (32, "nightstand"),
    (33, "toilet"),
    (34, "sink"),
    (35, "lamp"),
    (36, "bathtub"),
    (37, "bag"),
    (38, "other structure"),
    (39, "other furniture"),
    (40, "other properties"),
)


while True:
# for i in range(1):
    scene_id = SCENE_LIST[randint(0, len(SCENE_LIST) - 1)].strip('.ply')
    tex_id = [randint(0, len(TEXTURE) - 1) for _ in range(40)]

    prompt = f"{','.join([f'{TEXTURE[tex_id[idx]]} {CLASS_LABELS[idx][1]}' for idx in range(40)])}"        
    name = f"{randint(1, 1048576)}"

    command = COMMAND.replace('$NAME$', str(name)).replace('$PROMPT$', str(prompt)).replace('$SCENE_ID$', str(scene_id))

    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    # print(f"{scene_id}/{name}")
    # print(f"{command}")
    os.system(command)
# %%
