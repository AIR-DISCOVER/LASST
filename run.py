# %%
from random import shuffle, random, randint
from launch import launch
import sys
import os
from datetime import datetime

SCENE_LIST = os.listdir('/home/tb5zhh/data/full/train')

DATE = '2022-02-12'
# 2022-02-07: randomly choose: scene_id, label_id, texture, one class in one scene at a time e.g. "a rusted door"
# 2022-02-10: new textures, and keep only the textures as prompt e.g. "Nebula"
# 2022-02-12: adopt new textures and apply those to all classes in a scene

COMMAND = f'python sem_seg_main.py \
        --run branch \
        --obj_path $SCENE_ID$ \
        --output_dir \"results/batch/{DATE}/$SCENE_ID$/$NAME$\" \
        --prompt \"$PROMPT$\" \
        --label 1 2 3 4 5 6 7 8 9 10 11 12 \
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
        --background 0.5 0.5 0.5 \
        --rand_background \
        --frontview_center 1.8707 0.6303 \
        --with_prior_color \
        --normratio 0.05 \
        --color_only \
        --render_all_grad_one \
        --focus_one_thing'

TEXTURE = [
    'Milky Way',
    'clouds at sunset',
    'snow mountain',
    'blue Sky',
    'zebra',
    'van Gogh',
    'blue sea'
]

CLASS_LABELS = ('null', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture',)

n2i = lambda n, set: (n // (len(set)**3) % len(set), n //
                      (len(set)**2) % len(set), n //
                      (len(set)) % len(set), n % len(set))

from IPython import embed
while True:
    # label_id = randint(1, len(CLASS_LABELS) - 2)
    scene_id = SCENE_LIST[randint(0, len(SCENE_LIST) - 1)].strip('.ply')
    # if random() > 0.5:
    tex_id = [randint(0, len(TEXTURE) - 1) for _ in range(12)]
    # prompt = f"a {TEXTURE[tex_id]} {CLASS_LABELS[label_id]}"

    prompt = f"{','.join([TEXTURE[i] for i in tex_id])}"
    # else:
    #     env_id = randint(0, len(ENVS) - 1)
    #     prompt = f"a {CLASS_LABELS[label_id]} {ENVS[env_id]}"
        
    name = f"{prompt}"

    command = COMMAND.replace('$NAME$', str(name)).replace('$PROMPT$', str(prompt)).replace('$SCENE_ID$', str(scene_id))
    if os.path.isdir(f'results/batch/{DATE}/{scene_id}/{name}'):
        continue
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    print(f"{scene_id}/{name}")
    # print(command, file=sys.stderr) 
    os.system(command)

exit(0)
texture_total = len(TEXTURE)**4
t_commands = []
for i in range(texture_total):
    idxs = n2i(i, TEXTURE)
    plist = [f'a {TEXTURE[idxs[k]]} {CLASS_LABELS[k]}' for k in range(4)]
    prompt = ', '.join(plist)
    name = '-'.join(('_'.join(i.split()) for i in plist))
    t_commands.append(
        COMMAND.replace('$NAME$', name).replace('$PROMPT$', prompt))
shuffle(t_commands)

s_commands = []
for scene in ENVS:
    plist = [f'a {CLASS_LABELS[k]} {scene}' for k in range(4)]
    prompt = ', '.join(plist)
    name = '-'.join(('_'.join(i.split()) for i in plist))
    s_commands.append(
        COMMAND.replace('$NAME$', name).replace('$PROMPT$', prompt))

# FIXME
commands = s_commands[4:] + t_commands
shuffle(t_commands)

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
