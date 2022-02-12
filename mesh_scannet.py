import torch
import utils
from utils import device
import copy
import numpy as np
import PIL
from plyfile import PlyData
DEVICE = torch.device("cuda:0")
from IPython import embed
class Mesh():
    """Load Mesh from ply files (from full and full_mesh)"""
    FULL_PATH = '/home/tb5zhh/data/full/train'
    FULL_MESH_PATH = '/home/tb5zhh/data/full_mesh/train'

    def __init__(self, scan_id, color=torch.tensor([0.0,0.0,1.0])) -> None:
        full_ply_path = f"{self.FULL_PATH}/{scan_id}.ply"
        full_mesh_ply_path = f"{self.FULL_MESH_PATH}/{scan_id}.ply"

        full_plydata = PlyData.read(full_ply_path)
        full_mesh_plydata = PlyData.read(full_mesh_ply_path)

        meshes = []
        for i in range(len(full_mesh_plydata['face'])):
            meshes.append(full_mesh_plydata['face'][i][0])
        self.faces : torch.Tensor = torch.as_tensor(np.stack(meshes)).to(DEVICE).to(torch.long)

        self.labels : torch.Tensor = torch.as_tensor(np.asarray(full_plydata['vertex']['label'])).to(DEVICE)
        self.colors : torch.Tensor = torch.as_tensor(np.stack((full_plydata['vertex']['red']/256, full_plydata['vertex']['green']/256, full_plydata['vertex']['blue']/256),axis=1)).to(DEVICE).to(dtype=torch.float)
        self.vertices : torch.Tensor = torch.as_tensor(np.stack((full_mesh_plydata['vertex']['x'],full_mesh_plydata['vertex']['y'],full_mesh_plydata['vertex']['z']),axis=1)).to(DEVICE)

        self.vertex_normals : torch.Tensor = None
        self.face_normals : torch.Tensor = None
        self.texture_map : torch.Tensor = None
        self.face_uvs : torch.Tensor = None
        self.set_mesh_color(color)

    def standardize_mesh(self,inplace=False):
        mesh = self if inplace else copy.deepcopy(self)
        return utils.standardize_mesh(mesh)

    def normalize_mesh(self,inplace=False):

        mesh = self if inplace else copy.deepcopy(self)
        return utils.normalize_mesh(mesh)

    def update_vertex(self,verts,inplace=False):

        mesh = self if inplace else copy.deepcopy(self)
        mesh.vertices = verts
        return mesh

    def set_mesh_color(self,color):
        self.texture_map = utils.get_texture_map_from_color(self,color)
        self.face_attributes = utils.get_face_attributes_from_color(self,color)

    def set_image_texture(self,texture_map,inplace=True):

        mesh = self if inplace else copy.deepcopy(self)

        if isinstance(texture_map,str):
            texture_map = PIL.Image.open(texture_map)
            texture_map = np.array(texture_map,dtype=np.float) / 255.0
            texture_map = torch.tensor(texture_map,dtype=torch.float).to(device).permute(2,0,1).unsqueeze(0)


        mesh.texture_map = texture_map
        return mesh

    def divide(self,inplace=True):

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

    def mask_mesh(self, text_label, normals=False):

        # focus on only one things with labels
        # FixMe: text_label should be fixed

        ver_mask = self.labels.eq(text_label)

        face_num = self.faces.shape[0]
        face_mask = torch.ones([face_num]).eq(1).to(device)
        for i in range(face_num):
            face_mask[i] = ver_mask[self.faces[i]].all()

        mesh = copy.deepcopy(self)
        mesh.vertices = self.vertices[ver_mask]
        if normals:
            mesh.vertex_normals = self.vertex_normals[ver_mask]

        if self.colors is not None:
            mesh.colors = self.colors[ver_mask]


        # create new indices
        old_indice_to_new = []
        new_indice_to_old = []
        new = 0
        for old in range(ver_mask.shape[0]):
            if ver_mask[old] == True:
                old_indice_to_new.append(new)
                new_indice_to_old.append(old)
                new = new + 1
            else:
                old_indice_to_new.append(-1)
        old_indice_to_new = torch.tensor(old_indice_to_new)
        new_indice_to_old = torch.tensor(new_indice_to_old)

        mesh.faces = self.faces[face_mask]
        mesh.faces = old_indice_to_new[mesh.faces].to(device)
        if normals:
            mesh.face_normals = self.face_normals[face_mask]
        mesh.face_attributes = self.face_attributes[0][face_mask].unsqueeze(0)

        return mesh, old_indice_to_new, new_indice_to_old

    def get_mask(self, text_label):

        ver_mask = self.labels.eq(text_label).to(torch.long).to(device)

        face_num = self.faces.shape[0]
        face_mask = torch.ones([face_num]).eq(1).to(device)
        for i in range(face_num):
            face_mask[i] = ver_mask[self.faces[i]].all()
        face_mask = face_mask.to(torch.long)
        return ver_mask, face_mask