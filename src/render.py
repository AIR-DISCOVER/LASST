from turtle import back
from mesh_scannet import Mesh
import kaolin as kal
from utils import (get_camera_from_view2, get_camera_from_inside_out)
import matplotlib.pyplot as plt
from utils import device
import torch
from random import randint
import numpy as np
from IPython import embed


class Renderer():

    def __init__(self, mesh='sample.obj', lights=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), camera=None, dim=(224, 224)):

        if camera is None:
            # changing fovyangle is equivalent to changing focal length
            fovyangle = np.pi / 3
            camera = kal.render.camera.generate_perspective_projection(fovyangle).to(device)

        self.lights = lights.unsqueeze(0).to(device)
        self.camera_projection = camera
        self.dim = dim

    def render_y_views(self, mesh, num_views=8, show=False, lighting=True, background=None, mask=False):

        faces = mesh.faces
        n_faces = faces.shape[0]

        azim = torch.linspace(0, 2 * np.pi, num_views + 1)[:-1]  # since 0 =360 dont include last element
        # elev = torch.cat((torch.linspace(0, np.pi/2, int((num_views+1)/2)), torch.linspace(0, -np.pi/2, int((num_views)/2))))
        elev = torch.zeros(len(azim))
        images = []
        masks = []
        rgb_mask = []

        if background is not None:
            face_attributes = [mesh.face_attributes, torch.ones((1, n_faces, 3, 1), device='cuda')]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=2).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(mesh.vertices.to(device),
                                                                                                       mesh.faces.to(device),
                                                                                                       self.camera_projection,
                                                                                                       camera_transform=camera_transform)

            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1], face_vertices_image, face_attributes,
                                                                                     face_normals[:, :, -1])
            masks.append(soft_mask)

            if background is not None:
                image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                background_mask[torch.where(mask == 0)] = background
                image = torch.clamp(image + background_mask, 0., 1.)

            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    # ax.imshow(images[i].permute(1,2,0).cpu().numpy())
                    # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        return images

    def render_single_view(self, mesh, elev=0, azim=0, show=False, lighting=True, background=None, radius=2, return_mask=False):
        # if mesh is None:
        #     mesh = self._current_mesh
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        if background is not None:
            face_attributes = [mesh.face_attributes, torch.ones((1, n_faces, 3, 1), device='cuda')]
        else:
            face_attributes = mesh.face_attributes

        camera_transform = get_camera_from_view2(torch.tensor(elev), torch.tensor(azim), r=radius).to(device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(mesh.vertices.to(device),
                                                                                                   mesh.faces.to(device),
                                                                                                   self.camera_projection,
                                                                                                   camera_transform=camera_transform)

        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1], face_vertices_image, face_attributes, face_normals[:, :,
                                                                                                                                                                                                 -1])

        # Debugging: color where soft mask is 1
        # tmp_rgb = torch.ones((224,224,3))
        # tmp_rgb[torch.where(soft_mask.squeeze() == 1)] = torch.tensor([1,0,0]).float()
        # rgb_mask.append(tmp_rgb)

        if background is not None:
            image_features, mask = image_features

        image = torch.clamp(image_features, 0.0, 1.0)

        if lighting:
            image_normals = face_normals[:, face_idx].squeeze(0)
            image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
            image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
            image = torch.clamp(image, 0.0, 1.0)

        if background is not None:
            background_mask = torch.zeros(image.shape).to(device)
            mask = mask.squeeze(-1)
            assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
            background_mask[torch.where(mask == 0)] = background
            image = torch.clamp(image + background_mask, 0., 1.)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(figsize=(89.6, 22.4))
                axs.imshow(image[0].cpu().numpy())
                # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        if return_mask == True:
            return image.permute(0, 3, 1, 2), mask
        return image.permute(0, 3, 1, 2)

    def render_uniform_views(self, mesh, num_views=8, show=False, lighting=True, background=None, mask=False, center=[0, 0], radius=2.0):

        # if mesh is None:
        #     mesh = self._current_mesh

        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        azim = torch.linspace(center[0], 2 * np.pi + center[0], num_views + 1)[:-1]  # since 0 =360 dont include last element
        elev = torch.cat((torch.linspace(center[1], np.pi / 2 + center[1], int((num_views + 1) / 2)), torch.linspace(center[1], -np.pi / 2 + center[1], int((num_views) / 2))))
        images = []
        masks = []
        background_masks = []

        if background is not None:
            face_attributes = [mesh.face_attributes, torch.ones((1, n_faces, 3, 1), device='cuda')]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=radius).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(mesh.vertices.to(device),
                                                                                                       mesh.faces.to(device),
                                                                                                       self.camera_projection,
                                                                                                       camera_transform=camera_transform)

            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1], face_vertices_image, face_attributes,
                                                                                     face_normals[:, :, -1])
            masks.append(soft_mask)

            # Debugging: color where soft mask is 1
            # tmp_rgb = torch.ones((224,224,3))
            # tmp_rgb[torch.where(soft_mask.squeeze() == 1)] = torch.tensor([1,0,0]).float()
            # rgb_mask.append(tmp_rgb)

            if background is not None:
                image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                background_mask[torch.where(mask == 0)] = background
                background_masks.append(background_mask)
                image = torch.clamp(image + background_mask, 0., 1.)

            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)
        if background is not None:
            background_masks = torch.cat(background_masks, dim=0).permute(0, 3, 1, 2)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    # ax.imshow(background_masks[i].permute(1,2,0).cpu().numpy())
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                    # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        return images

    def render_front_views(self, mesh, num_views=8, std=8, center_elev=0, center_azim=0, show=False, lighting=True, background=None, mask=False, return_views=False, rand_background=False):
        # Front view with small perturbations in viewing angle
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        elev = torch.cat((torch.tensor([center_elev]), torch.randn(num_views - 1) * np.pi / std + center_elev))
        azim = torch.cat((torch.tensor([center_azim]), torch.randn(num_views - 1) * 2 * np.pi / std + center_azim))
        images = []
        masks = []
        rgb_mask = []

        if background is not None:
            face_attributes = [mesh.face_attributes, torch.ones((1, n_faces, 3, 1), device='cuda')]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=2).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(mesh.vertices.to(device),
                                                                                                       mesh.faces.to(device),
                                                                                                       self.camera_projection,
                                                                                                       camera_transform=camera_transform)
            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1], face_vertices_image, face_attributes,
                                                                                     face_normals[:, :, -1])
            masks.append(soft_mask)

            # Debugging: color where soft mask is 1
            tmp_rgb = torch.ones((224, 224, 3))
            tmp_rgb[torch.where(soft_mask.squeeze() == 1)] = torch.tensor([1, 0, 0]).float()
            rgb_mask.append(tmp_rgb)

            if background is not None:
                image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                if rand_background:
                    background_mask[torch.where(mask == 0)] = background + torch.rand_like(background, device=background.device) / 3
                else:
                    background_mask[torch.where(mask == 0)] = background
                image = torch.clamp(image + background_mask, 0., 1.)
            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)
        rgb_mask = torch.cat(rgb_mask, dim=0)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                plt.show()

        if return_views == True:
            return images, elev, azim
        else:
            return images

    def render_prompt_views(self, mesh, prompt_views, center=[0, 0], background=None, show=False, lighting=True, mask=False):

        # if mesh is None:
        #     mesh = self._current_mesh

        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]
        num_views = len(prompt_views)

        images = []
        masks = []
        rgb_mask = []
        face_attributes = mesh.face_attributes

        for i in range(num_views):
            view = prompt_views[i]
            if view == "front":
                elev = 0 + center[1]
                azim = 0 + center[0]
            if view == "right":
                elev = 0 + center[1]
                azim = np.pi / 2 + center[0]
            if view == "back":
                elev = 0 + center[1]
                azim = np.pi + center[0]
            if view == "left":
                elev = 0 + center[1]
                azim = 3 * np.pi / 2 + center[0]
            if view == "top":
                elev = np.pi / 2 + center[1]
                azim = 0 + center[0]
            if view == "bottom":
                elev = -np.pi / 2 + center[1]
                azim = 0 + center[0]

            if background is not None:
                face_attributes = [mesh.face_attributes, torch.ones((1, n_faces, 3, 1), device='cuda')]
            else:
                face_attributes = mesh.face_attributes

            camera_transform = get_camera_from_view2(torch.tensor(elev), torch.tensor(azim), r=2).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(mesh.vertices.to(device),
                                                                                                       mesh.faces.to(device),
                                                                                                       self.camera_projection,
                                                                                                       camera_transform=camera_transform)

            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1], face_vertices_image, face_attributes,
                                                                                     face_normals[:, :, -1])
            masks.append(soft_mask)

            if background is not None:
                image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                background_mask[torch.where(mask == 0)] = background
                image = torch.clamp(image + background_mask, 0., 1.)
            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                    # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        if not mask:
            return images
        else:
            return images, masks

    def find_appropriate_view(self, mesh, lower=0.6, upper=0.9, percent=1.):
        face_attributes = [mesh.face_attributes, torch.ones((1, mesh.faces.shape[0], 3, 1), device='cuda')]
        length = 1000
        for i in range(length):
            fov_alpha = i/length
            if i % 500 == 0:
                print(i)
            elev = torch.rand(1) * np.pi / 2
            azim = torch.rand(1) * 2 * np.pi
            fov = torch.clamp(np.pi / 2 * (1 - torch.normal(0., np.pi / 6, (1,)).abs()) * percent, 0, np.pi)
            camera_transform = get_camera_from_inside_out(elev, azim, r=1.0).to(device)
            camera_projection = kal.render.camera.generate_perspective_projection(fov).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                mesh.vertices.to(device),
                mesh.faces.to(device),
                camera_projection,
                camera_transform=camera_transform,
            )
            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1],
                self.dim[0],
                face_vertices_camera[:, :, :, -1],
                face_vertices_image,
                face_attributes,
                face_normals[:, :, -1],
            )
            # image_features, mask = image_features
            # mask = mask.squeeze(-1)
            # soft_mask = mask == 0
            # ratio = 1 - soft_mask.sum() / (224 * 224)
            ratio = torch.count_nonzero(face_idx[0] != -1) / (224*224)
            if ratio > lower and ratio < upper:
                break
        if i >= length - 1:
            print('retry')
            return None
        return None, elev, azim, fov

    def render_center_out_views(self,
                                mesh,
                                num_views=8,
                                elev_std=12,
                                azim_std=6,
                                show=False,
                                lighting=True,
                                background=None,
                                mask=False,
                                return_views=False,
                                rand_background=False,
                                fixed=True,
                                fixed_all=True,
                                render_args=None,
                                ini_camera_up_direction=False):
        """
            camera view from inside out
        """
        faces = mesh.faces
        n_faces = faces.shape[0]

        images = []
        masks = []
        rgb_mask = []
        ratios = []
        fovs = []

        if background is not None:
            face_attributes = [mesh.face_attributes, torch.ones((1, n_faces, 3, 1), device='cuda')]
        else:
            face_attributes = mesh.face_attributes
        for i in range(num_views):
            # j = i
            j = randint(0,len(render_args)-1)
            if fixed or fixed_all:
                elev = render_args[j][1]
                azim = render_args[j][2]
                fov = render_args[j][3]
            else:
                elev = torch.clamp(torch.normal(mean=render_args[j][1], std=elev_std * np.pi), 0, np.pi)
                azim = torch.clamp(torch.normal(mean=render_args[j][2], std=azim_std * np.pi), 0, 2 * np.pi)
                fov = torch.clamp(render_args[j][3] * 1 / torch.normal(mean=1, std=0.1, size=(1,)), 0, np.pi * 2 / 3)

            fov = np.pi/3
            fovs.append(fov/np.pi)

            camera_transform = get_camera_from_inside_out(elev, azim, r=1.0, ini_camera_up_direction=ini_camera_up_direction).to(device)
            camera_projection = kal.render.camera.generate_perspective_projection(fov).to(device)

            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(mesh.vertices.to(device),
                                                                                                       mesh.faces.to(device),
                                                                                                       camera_projection,
                                                                                                       camera_transform=camera_transform)
            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1],
                self.dim[0],
                face_vertices_camera[:, :, :, -1],
                face_vertices_image,
                face_attributes,
                face_normals[:, :, -1],
            )

            ratio = torch.count_nonzero(face_idx[0] != -1) / (224*224)
            ratios.append(ratio)

            if background is not None:
                image_features, mask = image_features

            masks.append(soft_mask)

            tmp_rgb = torch.ones((224, 224, 3))
            tmp_rgb[torch.where(soft_mask.squeeze() == 1)] = torch.tensor([1, 0, 0]).float()
            rgb_mask.append(tmp_rgb)

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                if rand_background:
                    rand = (torch.rand_like(background_mask, device=background.device) - 0.5) / 5
                    background_mask[torch.where(mask == 0)] = background + rand[torch.where(mask == 0)]
                else:
                    background_mask[torch.where(mask == 0)] = background
                image = torch.clamp(image + background_mask, 0., 1.)

            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)
        rgb_mask = torch.cat(rgb_mask, dim=0)
        ratios = torch.tensor(ratios)
        fovs = torch.tensor(fovs)

        # np.savetxt(f"render_ratio.txt", ratios.cpu())
        # np.savetxt(f"render_fov.txt", fovs.cpu())

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                plt.show()

        if return_views == True:
            return images, elev, azim
        else:
            return images
