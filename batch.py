from argparse import Namespace
import shutil
from sem_seg_main import run
import os
import json
from datetime import datetime
from random import randint
from pathlib import Path

VERSION = 'm1'
SCENE_LIST = os.listdir('/home/tb5zhh/data/full/train')

def main():
    while True:
        date = datetime.today().strftime('%Y-%m-%d') + VERSION
        with open('class.json') as f2, open('valid.json') as f3:
            class_labels = json.load(f2)
            valid_cate = json.load(f3)

        scene_id = SCENE_LIST[randint(0, len(SCENE_LIST) - 1)].strip('.ply')
        output_dir = f"results/batch/{date}/{scene_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        idx = 0
        while (Path(output_dir) / f'{idx}').exists():
            idx += 1
        (Path(output_dir) / f'{idx}').mkdir(parents=True, exist_ok=False)

        for config in sorted(os.listdir('configs')):
            try:
                with open(f'configs/{config}') as f:
                    args = json.load(f)

                args = Namespace(**args)
                args.dry_run = False
                args.obj_path = scene_id
                args.output_dir = output_dir

                prompt_idx = [randint(0, len(class_labels[i][2]) - 1) for i in valid_cate]
                args.prompt = ','.join([f"{class_labels[i][2][prompt_idx[idx]]} {class_labels[i][1]}" for idx, i in enumerate(valid_cate)]).split(' ')
                args.label = [class_labels[i][0] for i in valid_cate]


                args.dir = f"{args.output_dir}/{idx}/{args.name}"
                Path(args.dir).mkdir(parents=True, exist_ok=True)
                with open(Path(args.dir) / 'config.json', 'w') as f:
                    json.dump(vars(args), f, indent=4)
                run(args)
            except KeyboardInterrupt:
                shutil.rmtree(f"{args.output_dir}/{idx}/{args.name}")
                if len(os.listdir(f"{args.output_dir}/{idx}")) == 0:
                    shutil.rmtree(f"{args.output_dir}/{idx}")
                if len(os.listdir(f"{args.output_dir}")) == 0:
                    shutil.rmtree(f"{args.output_dir}")
                return 
            except:
                print(f"error with config {config}")
                shutil.rmtree(f"{args.output_dir}/{idx}/{args.name}")
                if len(os.listdir(f"{args.output_dir}/{idx}")) == 0:
                    shutil.rmtree(f"{args.output_dir}/{idx}")
                if len(os.listdir(f"{args.output_dir}")) == 0:
                    shutil.rmtree(f"{args.output_dir}")
        with open('list.log', 'a') as f:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t{scene_id} - {idx}", file=f)

if __name__ == '__main__':
    main()