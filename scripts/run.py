# %%
from random import shuffle, random, randint, choice, seed
from launch import launch
import sys
import os
from datetime import datetime
from plyfile import PlyData
from local import FULL_PATH

SCENE_LIST = os.listdir(FULL_PATH)

DATE = datetime.today().strftime('%Y-%m-%d')
# 2022-02-07: randomly choose: scene_id, label_id, texture, one class in one scene at a time e.g. "a rusted door"
# 2022-02-10: new textures, and keep only the textures as prompt e.g. "Nebula"
# 2022-02-12: adopt new textures and apply those to all classes in a scene
# 2022-03-01: fix the big bug of incorrespondance between class and label

COMMAND = f"""python main.py \
        --run branch \
        --obj_path $SCENE_ID$ \
        --label $LABEL$\
        --prompt \"$PROMPT$\" \
        --forbidden \"human face,English alphabet,lighting,human\"\
        --output_dir \"results/batch/{DATE}_one_one/$SCENE_ID$/$NAME$\" \
        --learning_rate 0.0005 \
        --lr_decay 0.9 \
        --n_iter 700 \
\
        --frontview_elev_std 0.01 \
        --frontview_azim_std 0.1 \
        --background 0.1 0.1 0.1 \
        --render_all_grad_all \
\
        --n_normaugs 1 \
        --n_augs 1 \
        --n_views 5 \
        --mincrop 0.6 \
        --maxcrop 0.9 \
        --view_min 0.25 \
        --view_max 0.7 \
        --normmincrop 0.6 \
        --normmaxcrop 0.9 \
\
        --color_only\
        --with_prior_color\
\
        --clipavg \
        --report_step 100 
"""

# %%
# ini_camera_up_direction

COLORS = [
    'ash grey',
    'eton blue',
    'celadon grass',
    'mint grass',
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
    'oak',
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
    (1, "wall",     ['brick', 'oak']),
    (2, "floor",    ['marble', 'oak', 'steel']),
    (3, "cabinet",  ['steel', 'oak',]),
    (4, "bed",      ['leather', 'silk']),
    (5, "chair",    ['oak', 'leather',]),
    (6, "sofa",     ['leather', 'fabric', 'silk']),
    (7, "table",    ['oak', 'steel', 'stone']),
    (8, "door",     ['oak', 'stone']),
    (9, "window",   ['oak',]),
    (10, "bookshelf",   ['oak', 'steel', 'stone']),
    (11, "picture",     ['Van Gogh\'s', 'rainbow',]),
    (12, "counter",     ['steel', 'oak',]),
    (13, "blinds",      ['fabric', 'silk',]),
    (14, "desk",        ['oak', 'steel',]),
    (15, "shelves",     ['oak', 'steel',]),
    (16, "curtain",     ['fabric', 'rainbow',]),
    (17, "dresser",     ['oak', 'stone', 'steel',]),
    (18, "pillow",      ['leather','silk', 'fabric']),
    (19, "mirror",      []),
    (20, "floor mat",   ['colorful', 'flaming', 'flowers', 'grass', 'rainbow',]),
    (21, "clothes",     ['colorful', 'flaming', 'flowers', 'grass',]),
    (22, "ceiling",     []),
    (23, "books",       ['colorful', 'flaming', 'flowers', 'grass',]),
    (24, "refridgerator", ['steel',]),
    (25, "television",  ['outdated',]),
    (26, "paper",       []),
    (27, "towel",       ['colorful', 'flaming', 'flowers', 'grass', 'rainbow',]),
    (28, "shower curtain", ['colorful', 'rainbow',]),
    (29, "box",         []),
    (30, "whiteboard",  []),
    (31, "person",      []),
    (32, "nightstand",  ['oak', 'steel',]),
    (33, "toilet",      ['golden',]),
    (34, "sink",        ['steel',]),
    (35, "lamp",        []),
    (36, "bathtub",     ['steel', 'golden', 'Chocolate']),
    (37, "bag",         []),
    (38, "other structure", []),
    (39, "other furniture", []),
    (40, "other properties",[]),
)

seed(42)

while True:
    # for i in range(1):
    scene_id = SCENE_LIST[randint(0, len(SCENE_LIST) - 1)].strip('.ply')

    a = PlyData.read(FULL_PATH+f"/{scene_id}.ply")
    labels = []
    for i in a.elements[0]['label']:
        if i not in labels:
            labels.append((i))
    shuffle(labels)

    choose_num = 0
    label_id = []
    for i in labels:
        if i not in [0, 9, 17,19, 21,22, 23, 25, 26, 27, 29, 30, 31, 35, 36, 37, 38, 39, 40]:
            label_id.append(i)
            choose_num = choose_num + 1
        if choose_num > 1:
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
        command = command.replace('$NAME$', str(name) + '/hsv_02/')
    elif sys.argv[1] == '1':    
        command = command.replace('$NAME$', str(name) + '/rgb_02/')
    elif sys.argv[1] == '2':    
        command = command.replace('$NAME$', str(name) + '/base')
    elif sys.argv[1] == '3':    
        command = command.replace('$NAME$', str(name) + '/text2mesh_angel')
    else:
        raise Exception("should add an arg number like: python run.py 3")

    # print(f"{scene_id}/{name}")
    # print(f"{command}")
    os.system(command)
# %%
