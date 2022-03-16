from argparse import Namespace
from sem_seg_main import run
import os
import json
from datetime import datetime
from random import randint
from pathlib import Path

SCENE_LIST = os.listdir('/home/tb5zhh/data/full/train')

while True:
    date = datetime.today().strftime('%Y-%m-%d')
    with open('base.json') as f1, open('class.json') as f2, open('valid.json') as f3:
        args = json.load(f1)
        class_labels = json.load(f2)
        valid_cate = json.load(f3)

    args = Namespace(args)
    args.dry_run = False

    scene_id = SCENE_LIST[randint(0, len(SCENE_LIST) - 1)].strip('.ply')
    args.obj_path = scene_id
    args.output_dir = f"{args.output_dir}/{scene_id}"

    idx = 0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    while (Path(args.output_dir) / f'{idx}').exists():
        idx += 1
    (Path(args.output_dir) / f'{idx}').mkdir(parents=True, exist_ok=False)
    args.dir = f"{args.output_dir}/{idx}"

    prompt_idx = [randint(0, len(class_labels[i][2]) - 1) for i in valid_cate]
    args.prompt = ','.join([f"{class_labels[i][2][prompt_idx[i]]} {class_labels[i][1]}" for i in valid_cate])
    args.label = ' '.join([class_labels[i][0] for i in valid_cate])
    run(args)