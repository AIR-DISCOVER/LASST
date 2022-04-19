# %%
from pickletools import int4
from random import shuffle, random, randint, choice, seed
from launch import launch
import sys
import os
from datetime import datetime
from plyfile import PlyData
import numpy as np

SCENE_LIST = os.listdir('/home/tb5zhh/data/full/test')

DATE = datetime.today().strftime('%Y-%m-%d')
# 2022-02-07: randomly choose: scene_id, label_id, texture, one class in one scene at a time e.g. "a rusted door"
# 2022-02-10: new textures, and keep only the textures as prompt e.g. "Nebula"
# 2022-02-12: adopt new textures and apply those to all classes in a scene
# 2022-03-01: fix the big bug of incorrespondance between class and label

pred_label_path = str(100)

COMMAND = f"""
python sem_seg_main.py \
        --obj_path $SCENE_ID$ \
        --label $LABEL$\
        --prompt \"$PROMPT$\" \
        --forbidden \"human face,English alphabet,lighting,human\"\
        --output_dir \"results/batch/{DATE}/$SCENE_ID$/$NAME$\" \
        --learning_rate 0.001\
        --lr_decay 0.9 \
        --n_iter 1000\
        --frontview_elev_std 0.01\
        --frontview_azim_std 0.1\
        --background 0.1 0.1 0.1\
        --render_one_grad_one\
        --n_normaugs 3\
        --n_augs 3\
        --n_views 8\
        --mincrop 0.6\
        --maxcrop 0.9\
        --view_min 0.5\
        --view_max 0.7\
        --normmincrop 0.6\
        --normmaxcrop 0.9\
        --color_only\
        --with_prior_color\
        --clipavg\
        --sv_stat_loss_weight 0.1\
        --report_step 100
"""

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
    'brick',
    'wooden',
    'marble',
    #'plastic',
    'steel',
    'leather',
    'diamond texture',
    #'fabric',
    'icy',
    #'ceramic'
]

CLASS_LABELS = (
    (0, "null",     []),
    (1, "wall",     ['brick', 'wooden', 'icy',]),
    (2, "floor",    ['marble', 'icy', 'wooden', 'steel']),
    (3, "cabinet",  ['steel', 'wooden',]),
    (4, "bed",      ['leather', 'diamond texture', ]),
    (5, "chair",    ['wooden', 'leather', ]),
    (6, "sofa",     ['leather', 'fabric',]),
    (7, "table",    ['wooden', 'steel',]),
    (8, "door",     ['wooden',]),
    (9, "window",   ['wooden',]),
    (10, "bookshelf",   ['wooden', 'steel',]),
    (11, "picture",     ['Van Gogh', 'rainbow',]),
    (12, "counter",     ['steel', 'wooden',]),
    (13, "blinds",      ['fabric', 'rainbow',]),
    (14, "desk",        ['wooden', 'stone', 'steel',]),
    (15, "shelves",     ['wooden', 'Chocolate', 'steel',]),
    (16, "curtain",     ['fabric', 'rainbow',]),
    (17, "dresser",     ['wooden', 'stone', 'steel',]),
    (18, "pillow",      ['leather',]),
    (19, "mirror",      []),
    (20, "floor mat",   ['blue', 'red', 'yellow', 'green', 'rainbow',]),
    (21, "clothes",     ['blue', 'red', 'yellow', 'green',]),
    (22, "ceiling",     []),
    (23, "books",       ['blue', 'red', 'yellow', 'green',]),
    (24, "refridgerator", ['steel', 'red',]),
    (25, "television",  ['outdated',]),
    (26, "paper",       []),
    (27, "towel",       ['blue', 'red', 'yellow', 'green', 'rainbow',]),
    (28, "shower curtain", ['blue', 'red', 'yellow', 'green', 'rainbow',]),
    (29, "box",         []),
    (30, "whiteboard",  ['math problems in',]),
    (31, "person",      []),
    (32, "nightstand",  ['wooden', 'marble',]),
    (33, "toilet",      ['golden', 'Chocolate',]),
    (34, "sink",        ['steel', 'Chocolate']),
    (35, "lamp",        []),
    (36, "bathtub",     ['steel', 'golden', 'Chocolate']),
    (37, "bag",         []),
    (38, "other structure", []),
    (39, "other furniture", []),
    (40, "other properties",[]),
)

for i in range(len(SCENE_LIST)):
    # for i in range(1):
    scene_id = SCENE_LIST[i].strip('.ply')

    a = np.loadtxt(f"/home/jinbu/text2mesh/data/pred_label/{pred_label_path}/{scene_id}.txt", dtype=int)
    labels = []
    for i in a:
        if i not in labels:
            labels.append((i))
    shuffle(labels)

    choose_num = 0
    label_id = [1, 2]
    for i in labels:
        if i not in [0, 1, 2, 9, 19, 22, 26, 29, 31, 35, 37, 38, 39, 40]:
            label_id.append(i)
            choose_num = choose_num + 1
        if choose_num > 4:
            break

    prompt = f"{','.join([f'{choice(CLASS_LABELS[id][2])} {CLASS_LABELS[id][1]}' for id in label_id])}"        
    name = "_".join([str(i) for i in label_id])
    chose_label = " ".join(str(i) for i in label_id)

    command = COMMAND.replace('$PROMPT$', str(prompt)).replace('$SCENE_ID$', str(scene_id)).replace('$LABEL$', chose_label)

    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    """
    if sys.argv[1] == '0':    # init
        command = command.replace('$NAME$', str(name) + '/init')
        command = command + " --lighting"
    elif sys.argv[1] == '1':    # random_focal
        command = command.replace('$NAME$', str(name) + '/random_focal')
        command = command + " --rand_focal --lighting"
    elif sys.argv[1] == '2':    # initial_color
        command = command.replace('$NAME$', str(name) + '/initial_color')
        command = command + " --with_prior_color"
    elif sys.argv[1] == '3':    # hsv
        command = command.replace('$NAME$', str(name) + '/hsv')
        command = command + " --with_hsv_loss --lighting"
    else:
        raise Exception("should add an arg number like: python run.py 3")
    """

    if sys.argv[1] == '0':    
        command = command.replace('$NAME$', str(name) + '/base')
    elif sys.argv[1] == '1':    
        command = command.replace('$NAME$', str(name) + '/cuda_1')
    elif sys.argv[1] == '2':    
        command = command.replace('$NAME$', str(name) + '/cuda_2')
    elif sys.argv[1] == '3':    
        command = command.replace('$NAME$', str(name) + '/rander_all_grad_all')
    else:
        raise Exception("should add an arg number like: python run.py 3")

    # print(f"{scene_id}/{name}")
    # print(f"{command}")
    os.system(command)
# %%
