import os

base_path = "./results/batch/2022-03-09"

files = os.listdir(base_path)

for file in files:
    scene_path = os.path.join(base_path, file)
    print(f"scene_path: {scene_path}")
    scenes = os.listdir(scene_path)
    if not scenes:
        os.rmdir(scene_path)
    for scene in scenes:
        exps_path = os.path.join(scene_path, scene)
        print(f"exps_path: {exps_path}")
        exps = os.listdir(exps_path)
        if not exps:
            os.rmdir(exps_path)
            continue
        for exp in exps:
            final_path = os.path.join(exps_path, exp)
            print(f"final_path: {final_path}")
            obj_and_txts = os.listdir(final_path)
            found_obj = 0
            for obj_and_txt in obj_and_txts:
                if ".obj" in obj_and_txt:
                    found_obj = 1
            if found_obj == 0:
                os.rmdir(final_path)

