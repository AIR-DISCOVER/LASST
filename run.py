# %%
from random import shuffle
from launch import launch
import sys
import os
from datetime import datetime

COMMAND = f'python sem_seg_main.py \
        --run branch \
        --obj_path data/scene0002_00/scene0002_00_vh_clean_2.ply \
        --output_dir results/batch/scene0002_00/{datetime.today().strftime("%Y-%m-%d")}/$NAME$ \
        --prompt \"$PROMPT$\" \
        --label 2 5 6 8\
        --sigma 5.0  \
        --clamp tanh \
        --n_normaugs 4 \
        --n_augs 1 \
        --normmincrop 0.1 \
        --normmaxcrop 0.1 \
        --geoloss \
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
        --n_iter 1000 \
        --learning_rate 0.0005 \
        --normal_learning_rate 0.0005 \
        --background 1 1 1 \
        --frontview_center 1.8707 0.6303 \
        --lighting \
        --normratio 0.05 \
        --color_only \
        --render_all_grad_one \
        --focus_one_thing'

TEXTURE = [
    'wooden',
    'painted',
    'glass',
    'metal',
    'brick',
    'plastic',
    'shiny',
    'rusted',
]

SCENES = [
    'on the beach',
    'in the forest',
    'under the sea',
    'in the sky',
    'in the rain',
    'during sunset',
    'at the midnight',
]

CLS_NAME = ['floor', 'chair', 'sofa', 'door']
# 2 floor
# 5 chair
# 6 sofa
# 8 door

n2i = lambda n, set: (n // (len(set)**3) % len(set), n //
                      (len(set)**2) % len(set), n //
                      (len(set)) % len(set), n % len(set))

texture_total = len(TEXTURE)**4
t_commands = []
for i in range(texture_total):
    idxs = n2i(i, TEXTURE)
    plist = [f'a {TEXTURE[idxs[k]]} {CLS_NAME[k]}' for k in range(4)]
    prompt = ', '.join(plist)
    name = '-'.join(('_'.join(i.split()) for i in plist))
    t_commands.append(
        COMMAND.replace('$NAME$', name).replace('$PROMPT$', prompt))
shuffle(t_commands)

s_commands = []
for scene in SCENES:
    plist = [f'a {CLS_NAME[k]} {scene}' for k in range(4)]
    prompt = ', '.join(plist)
    name = '-'.join(('_'.join(i.split()) for i in plist))
    s_commands.append(
        COMMAND.replace('$NAME$', name).replace('$PROMPT$', prompt))

# FIXME
commands = s_commands[4:] + t_commands

card = sys.argv[1].split(',')
envss = [os.environ.copy() for i in range(len(card))]
for i in range(len(envss)):
    envss[i]['CUDA_VISIBLE_DEVICES'] = card[i]

step = len(commands) // int(sys.argv[3]) + 1
start = int(sys.argv[2]) * step
end = (int(sys.argv[2]) + 1) * step


for i in range(start, end):
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    # print(commands[i])
    os.system(commands[i])
    # launch(lambda s: commands[i].split(), env=lambda s: envss[0])