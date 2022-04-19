
import os
from plyfile import PlyData

SCENE_LIST = os.listdir('/home/tb5zhh/data/full_mesh/train')


train_num = 0
all_ver_num = 0
all_face_num = 0

for scene_id in SCENE_LIST:
    a = PlyData.read(f"/home/tb5zhh/data/full_mesh/train/{scene_id}")
    vertices_num = len(a.elements[0])
    faces_num = len(a.elements[1])

    train_num += 1
    all_ver_num += vertices_num
    all_face_num += faces_num

all_ver_num = all_ver_num/train_num
all_face_num = all_face_num/train_num

print(f"mesh_num:{train_num}")
print(f"ave_ver_num:{all_ver_num}")
print(f"ave_face_num:{all_face_num}")


