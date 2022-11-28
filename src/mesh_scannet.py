import torch
import utils
from utils import device
import copy
import numpy as np
import PIL
from plyfile import PlyData
import kaolin.ops.mesh

DEVICE = torch.device("cuda:0")
from IPython import embed
from local import FULL_PATH, FULL_MESH_PATH, TEST_FULL_PATH, TEST_FULL_MESH_PATH
from tqdm import tqdm
class Mesh():
    """Load Mesh from ply files (from full and full_mesh)"""
    # FULL_PATH = '/home/tb5zhh/data/full/train'
    # FULL_MESH_PATH = '/home/tb5zhh/data/full_mesh/train'

    def __init__(self, scan_id, pred_label_path=None, color=torch.tensor([0.0, 0.0, 1.0]), setup=True) -> None:
        if not setup:
            return
        if pred_label_path is None:
            assert int(scan_id.split("scene")[-1].split("_")[0]) < 707
            full_ply_path = f"{FULL_PATH}/{scan_id}.ply"
            full_mesh_ply_path = f"{FULL_MESH_PATH}/{scan_id}.ply"

            full_plydata = PlyData.read(full_ply_path) # -> v2, rgb
            full_mesh_plydata = PlyData.read(full_mesh_ply_path) # -> v2.label.ply, with vertex, face, label

            meshes = []
            for i in range(len(full_mesh_plydata['face'])):
                meshes.append(full_mesh_plydata['face'][i][0])
            self.faces: torch.Tensor = torch.as_tensor(np.stack(meshes)).to(DEVICE).to(torch.long)

            self.labels: torch.Tensor = torch.as_tensor(np.asarray(full_mesh_plydata['vertex']['label']).astype(np.int16)).to(DEVICE)

            self.colors: torch.Tensor = torch.as_tensor(np.stack((full_plydata['vertex']['red'] / 256, full_plydata['vertex']['green'] / 256, full_plydata['vertex']['blue'] / 256),
                                                                axis=1)).to(DEVICE).to(dtype=torch.float)
            self.vertices: torch.Tensor = torch.as_tensor(np.stack((full_mesh_plydata['vertex']['x'], full_mesh_plydata['vertex']['y'], full_mesh_plydata['vertex']['z']), axis=1)).to(DEVICE)

            self.vertex_normals: torch.Tensor = None
            self.face_normals: torch.Tensor = None
            self.texture_map: torch.Tensor = None
            self.face_attributes: torch.Tensor = None
            self.face_uvs: torch.Tensor = None

            self.texture_map = utils.get_texture_map_from_color(self, color)
            # self.face_attributes = utils.get_face_attributes_from_color(self, color)  # FIXME0
            self.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(self.colors.unsqueeze(0), self.faces).squeeze().unsqueeze(0)
        else:   # use predicted label instead of ground truth
            full_ply_path = f"{FULL_PATH}/{scan_id}.ply"
            full_mesh_ply_path = f"{FULL_MESH_PATH}/{scan_id}.ply"

            full_plydata = PlyData.read(full_ply_path)
            full_mesh_plydata = PlyData.read(full_mesh_ply_path)

            meshes = []
            for i in range(len(full_mesh_plydata['face'])):
                meshes.append(full_mesh_plydata['face'][i][0])
            self.faces: torch.Tensor = torch.as_tensor(np.stack(meshes)).to(DEVICE).to(torch.long)

            self.labels: torch.Tensor = torch.as_tensor(np.loadtxt(pred_label_path)).to(DEVICE)
            


            self.colors: torch.Tensor = torch.as_tensor(np.stack((full_plydata['vertex']['red'] / 256, full_plydata['vertex']['green'] / 256, full_plydata['vertex']['blue'] / 256),
                                                                axis=1)).to(DEVICE).to(dtype=torch.float)
            self.vertices: torch.Tensor = torch.as_tensor(np.stack((full_mesh_plydata['vertex']['x'], full_mesh_plydata['vertex']['y'], full_mesh_plydata['vertex']['z']), axis=1)).to(DEVICE)

            self.vertex_normals: torch.Tensor = None
            self.face_normals: torch.Tensor = None
            self.texture_map: torch.Tensor = None
            self.face_attributes: torch.Tensor = None
            self.face_uvs: torch.Tensor = None

            self.texture_map = utils.get_texture_map_from_color(self, color)
            # self.face_attributes = utils.get_face_attributes_from_color(self, color)  # FIXME0
            self.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(self.colors.unsqueeze(0), self.faces).squeeze().unsqueeze(0)


    def standardize_mesh(self, inplace=False):
        mesh = self if inplace else copy.deepcopy(self)
        return utils.standardize_mesh(mesh)

    def normalize_mesh(self, inplace=False):

        mesh = self if inplace else copy.deepcopy(self)
        return utils.normalize_mesh(mesh)

    def update_vertex(self, verts, inplace=False):

        mesh = self if inplace else copy.deepcopy(self)
        mesh.vertices = verts
        return mesh

    def set_image_texture(self, texture_map, inplace=True):

        mesh = self if inplace else copy.deepcopy(self)

        if isinstance(texture_map, str):
            texture_map = PIL.Image.open(texture_map)
            texture_map = np.array(texture_map, dtype=np.float) / 255.0
            texture_map = torch.tensor(texture_map, dtype=torch.float).to(device).permute(2, 0, 1).unsqueeze(0)

        mesh.texture_map = texture_map
        return mesh

    def divide(self, inplace=True):

        mesh = self if inplace else copy.deepcopy(self)
        new_vertices, new_faces, new_face_uvs = utils.add_vertices(mesh)
        mesh.vertices = new_vertices
        mesh.faces = new_faces
        mesh.face_uvs = new_face_uvs
        return mesh

    def export(self, file, color=None):
        with open(file, "w+") as f:
            for vi, v in enumerate(self.vertices):
                if color is None:
                    f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
                else:
                    f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
                if self.vertex_normals is not None:
                    f.write("vn %f %f %f\n" % (self.vertex_normals[vi, 0], self.vertex_normals[vi, 1], self.vertex_normals[vi, 2]))
            for face in self.faces:
                f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))

    def mask_mesh(self, ver_mask, face_mask, old_indice_to_new, new_indice_to_old, normals=False):

        mesh = copy.deepcopy(self)
        mesh.vertices = self.vertices[ver_mask]
        if normals:
            mesh.vertex_normals = self.vertex_normals[ver_mask]

        if self.colors is not None:
            mesh.colors = self.colors[ver_mask]

        mesh.faces = self.faces[face_mask]
        mesh.faces = old_indice_to_new[mesh.faces].to(device)  # TODO check
        if normals:
            mesh.face_normals = self.face_normals[face_mask]
        mesh.face_attributes = self.face_attributes.squeeze()[face_mask].unsqueeze(0)

        return mesh

    def get_mask(self, text_label):

        ver_mask = self.labels.eq(text_label).to(device)

        # FIXME
        # selecting faces whose vertices all have the required label
        face_mask = ver_mask[self.faces].all(dim=-1)
        # create mapping between indices
        old_indice_to_new = (-1 * torch.ones_like(ver_mask))
        old_indice_to_new[ver_mask] = torch.tensor((range((ver_mask == 1).sum()))).cuda()
        new_indice_to_old = torch.tensor(range(len(ver_mask))).cuda()[ver_mask]
        assert len(old_indice_to_new) == len(ver_mask)
        assert len(new_indice_to_old) == (ver_mask == 1).sum()

        # old_indice_to_new = []
        # new_indice_to_old = []
        # print("start")
        # new = 0
        # for old in range(ver_mask.shape[0]):
        #     if ver_mask[old] == True:
        #         old_indice_to_new.append(new)
        #         new_indice_to_old.append(old)
        #         new = new + 1
        #     else:
        #         old_indice_to_new.append(-1)
        # old_indice_to_new = torch.tensor(old_indice_to_new)
        # new_indice_to_old = torch.tensor(new_indice_to_old)
        # embed()

        return ver_mask, face_mask, old_indice_to_new, new_indice_to_old

    def clone(self):
        new_mesh = Mesh(1, setup=False)
        new_mesh.faces = self.faces.detach() if self.faces is not None else None
        new_mesh.labels = self.labels.detach() if self.labels is not None else None
        new_mesh.colors = self.colors.detach() if self.colors is not None else None
        new_mesh.vertices = self.vertices.detach() if self.vertices is not None else None
        new_mesh.vertex_normals = self.vertex_normals.detach() if self.vertex_normals is not None else None
        new_mesh.face_normals = self.face_normals.detach() if self.face_normals is not None else None
        new_mesh.texture_map = self.texture_map.detach() if self.texture_map is not None else None
        new_mesh.face_attributes = self.face_attributes.detach() if self.face_attributes is not None else None
        new_mesh.face_uvs = self.face_uvs.detach() if self.face_uvs is not None else None

        new_mesh.texture_map = self.texture_map.detach() if self.texture_map is not None else None
        new_mesh.face_attributes = self.face_attributes.detach() if self.face_attributes is not None else None
        return new_mesh
    
    def replace_label_mask(self, pred_label_path):
        if not (self.labels == 0).all():
            raise ValueError("this mesh is not in test dataset")

        self.labels = self.labels

